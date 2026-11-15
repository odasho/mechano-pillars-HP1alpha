"""
Segmentation and shape analysis of one annotated nucleus and its HP1α condensates.

`ProcessImage` runs the full pipeline for one nucleus when it is called:

1. Load the multichannel z-stack (for the voxel size and, if pillars are present, the
   pillar channel), the Ilastik segmentation(s) and the hand-drawn nucleus annotation.
2. Crop the segmentations to the annotation and segment the nucleus, the HP1α
   condensates and (optionally) the pillars.
3. Measure the nucleus (volume, axis lengths, sphericity, flatness) and the condensates
   (number, volumes, axis lengths, radial distance to the nucleus centroid and minimal
   distance to the nuclear periphery).
4. Save voxel and physical coordinates of the nucleus, condensates, pillars and actin,
   which are used by downstream analysis.

Coordinates and sizes follow the (z, y, x) axis order of the image stack. Physical
units are µm.
"""

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import napari
import numpy as np
import pandas as pd
import pyclesperanto_prototype as cle
import SimpleITK as sitk
from aicsimageio import AICSImage
from napari.layers import Image, Shapes
from napari_crop._function import crop_region
from numpy import linalg as LA
from radiomics import shape
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops

from .utils import (
    axis_lengths,
    dotdict,
    find_coordinates,
    lmbda_dist_out_from_center,
    scale_coordinates,
)

# Classes in the 4-class Ilastik segmentation of the HP1α channel
NUCLEOPLASM, NUCLEOLI, HP1A_CONDENSATES = 2, 3, 4  # 1 = background
# Classes in the Ilastik segmentation of the actin channel
ACTIN_FIBERS, ACTIN_FOCI = 2, 3  # 1 = background


@dataclass
class Results:
    """
    Measurements of one nucleus.
    """

    # Metadata
    image_file: str = field(init=False, repr=False)
    annotation_file_nucleus: str = field(init=False, repr=False)
    annotation_file_actin: str = field(init=False, repr=False)
    segmentation_file: str = field(init=False, repr=False)
    cell_type: str = field(init=False, repr=False)
    condition: str = field(init=False, repr=False)
    adherence_time: str = field(init=False, repr=False)
    replicate_nr: str = field(init=False, repr=False)
    pillar_step: str = field(init=False, repr=False)

    # Nucleus
    nucleus_id: int = field(init=False, repr=False)
    voxels_nucleus: float = field(init=False, repr=False)  # number of voxels
    major_axis_nucleus: float = field(init=False, repr=False)  # µm
    intermediate_axis_nucleus: float = field(init=False, repr=False)  # µm
    minor_axis_nucleus: float = field(init=False, repr=False)  # µm
    sphericity: float = field(init=False, repr=False)
    flatness: float = field(init=False, repr=False)

    # Condensates (one entry per condensate)
    tot_num_condensates: int = field(init=False, repr=False)
    condensate_ids: list = field(init=False, repr=False)
    voxels_condensates: list = field(init=False, repr=False)  # number of voxels
    major_axis_HP1: list = field(init=False, repr=False)  # µm
    intermediate_axis_HP1: list = field(init=False, repr=False)  # µm
    minor_axis_HP1: list = field(init=False, repr=False)  # µm
    r_dist_condensates: list = field(init=False, repr=False)  # µm, condensate centroid to nucleus centroid
    p_minimal_distance_condensates: list = field(init=False, repr=False)  # µm, condensate centroid to periphery

    # Voxel size of the image stack
    physical_pixel_size_z: float = field(init=False, repr=False)  # µm
    physical_pixel_size_x: float = field(init=False, repr=False)  # µm
    physical_pixel_size_y: float = field(init=False, repr=False)  # µm
    pixel_size_um2: float = field(init=False, repr=False)
    voxel_size_um3: float = field(init=False, repr=False)

    def dict(self):
        """Return all fields as a dictionary."""
        return asdict(self)


class ProcessImage:
    """
    Segment and measure one annotated nucleus and its HP1α condensates.

    Calling an instance runs the pipeline described in the module docstring. The results
    are available afterwards as `df_img_results`, and the coordinate files are written
    to `coordinate_files`.

    Paths in `metadata` are used as follows: `image_file` is resolved relative to
    `data_dir`; `segmentation_file`, `annotation_file_nucleus` and the optional actin
    entries are resolved relative to the current working directory.
    """

    def __init__(
        self,
        metadata,
        data_dir,
        coordinate_files,
        meta_file,
        show_viewer=False,
        control=False,
        min_condensate_voxels=10,
    ):
        """
        Parameters
        ----------
        metadata : dict
            Contents of the metadata JSON of the nucleus: paths to the image, the Ilastik
            segmentation(s) and the annotation(s), and the experimental conditions
            (`cell_type`, `condition`, `adherence_time`, `replicate_nr`, `pillar_step`).
            Optional keys: `annotation_file_actin`, `segmentation_file_actin` and
            `pillar_threshold`.
        data_dir : Path
            Directory that `metadata["image_file"]` is relative to.
        coordinate_files : Path
            Directory where the coordinate files are saved.
        meta_file : Path
            Path to the metadata JSON. The nucleus number in the coordinate file names
            is read from its name (`..._nuc<N>.json`).
        show_viewer : bool, optional
            If True, show the images, segmentations and measurements in napari.
        control : bool, optional
            If True, skip pillar detection (images without pillars).
        min_condensate_voxels : int, optional
            Condensates smaller than this number of voxels are discarded as noise.
        """

        if data_dir is None:
            raise ValueError("The path to the data directory with images must be provided.")
        if not isinstance(data_dir, Path):
            raise TypeError("The path to the data directory must be provided as a pathlib.Path object.")
        if not isinstance(show_viewer, bool):
            raise TypeError("show_viewer must be provided as True or False")
        if not isinstance(control, bool):
            raise TypeError("control must be provided as True or False")
        if not isinstance(min_condensate_voxels, int) or min_condensate_voxels < 0:
            raise TypeError("min_condensate_voxels must be a non-negative integer")

        self._metadata = dotdict(metadata)  # allows metadata.key; missing keys give None
        self._data_dir = data_dir
        self._coordinate_files = coordinate_files
        self._meta_file = meta_file
        self._show_viewer = show_viewer
        self._control = control
        self._min_condensate_voxels = min_condensate_voxels
        self._results = Results()
        self._viewer_obj = None

    def __call__(self):
        """Run the pipeline. The results are stored in `df_img_results`."""

        self._process_metadata()

        # 1. Load and crop
        (
            channels,
            scaling_factors,
            annotation_nucleus,
            annotation_actin,
            shapes_layer_nucleus,
            shapes_layer_actin,
            cropped_cond,
            props_cond,
            cropped_nuc,
            props_nuc,
            cropped_actin_foci,
            props_actin_foci,
            cropped_actin_fibers,
            props_actin_fibers,
        ) = self._load_image()
        channel_DAPI, channel_HP1a, channel_actin, channel_pillar = channels

        # 2. Segment
        labeled_HP1 = self._segment_condensates(cropped_cond)
        labeled_nucleus = self._segment_nucleus(cropped_nuc)

        labeled_pillar = labeled_pillar_cropped = props_pillars = None
        if not self._control:
            # Pillars are cropped to the actin annotation if there is one, otherwise to
            # the nucleus annotation
            shapes_for_pillar = shapes_layer_actin if shapes_layer_actin is not None else shapes_layer_nucleus
            labeled_pillar, labeled_pillar_cropped, props_pillars = self._segment_pillars(
                channel_pillar, scaling_factors, shapes_for_pillar
            )

        # 3. Measure
        self._extract_condensate_properties(labeled_HP1)
        self._extract_nucleus_properties(labeled_nucleus)
        nucleus_coord, physical_coord_nucleus = self._extract_nucleus_axes(labeled_nucleus, props_nuc)
        self._extract_condensate_axes(labeled_HP1, scaling_factors)

        centroid_nucleus, physical_centroid_nucleus, centroid_conds_arr, physical_centroid_conds_arr = (
            self._extract_centroids(labeled_HP1, labeled_nucleus, props_nuc, props_cond)
        )
        periphery_coord_arr, physical_periphery_arr = self._define_nucleus_periphery(
            nucleus_coord, centroid_nucleus, labeled_nucleus, props_nuc
        )
        self._radial_distances(physical_centroid_nucleus, physical_centroid_conds_arr)
        coord_edge_p_cond = self._min_peripheral_distances(physical_centroid_conds_arr, physical_periphery_arr)

        # 4. Save coordinates for the downstream analyses (nucleus coordinates are saved
        # in _extract_nucleus_axes)
        self._extract_condensate_coordinates(labeled_HP1, props_cond)
        self._extract_actin_coordinates(cropped_actin_foci, props_actin_foci, cropped_actin_fibers, props_actin_fibers)
        filtered_pillars_voxel_v = filtered_pillars_physical_v = None
        if not self._control:
            filtered_pillars_voxel_v, filtered_pillars_physical_v = self._extract_pillar_coordinates(
                labeled_pillar_cropped, props_pillars
            )

        if self._show_viewer:
            self._viewer_obj = self._viewer(
                channel_DAPI=channel_DAPI,
                channel_HP1a=channel_HP1a,
                channel_actin=channel_actin,
                channel_pillar=channel_pillar,
                scaling_factors=scaling_factors,
                annotation_nucleus=annotation_nucleus,
                annotation_actin=annotation_actin,
                cropped_nuc=cropped_nuc,
                props_nuc=props_nuc,
                cropped_cond=cropped_cond,
                props_cond=props_cond,
                cropped_actin_foci=cropped_actin_foci,
                props_actin_foci=props_actin_foci,
                cropped_actin_fibers=cropped_actin_fibers,
                props_actin_fibers=props_actin_fibers,
                labeled_nucleus=labeled_nucleus,
                labeled_HP1=labeled_HP1,
                labeled_pillar=labeled_pillar,
                labeled_pillar_cropped=labeled_pillar_cropped,
                props_pillars=props_pillars,
                filtered_pillars_voxel_v=filtered_pillars_voxel_v,
                filtered_pillars_physical_v=filtered_pillars_physical_v,
                nucleus_coord=nucleus_coord,
                physical_coord_nucleus=physical_coord_nucleus,
                periphery_coord_arr=periphery_coord_arr,
                physical_centroid_nucleus=physical_centroid_nucleus,
                physical_centroid_conds_arr=physical_centroid_conds_arr,
                centroid_conds_arr=centroid_conds_arr,
                coord_edge_p_cond=coord_edge_p_cond,
            )

    @property
    def df_img_results(self):
        """Results as a table with one row per condensate."""
        return pd.DataFrame.from_dict(self._results.dict())

    @property
    def viewer_results(self):
        """The napari viewer, if `show_viewer=True`; otherwise None."""
        return self._viewer_obj

    # ------------------------------------------------------------------ load images

    def _process_metadata(self):
        """Copy the file names and experimental conditions from the metadata into the results."""

        self._results.image_file = Path(self._metadata.image_file).name
        self._results.annotation_file_nucleus = Path(self._metadata.annotation_file_nucleus).name
        self._results.segmentation_file = Path(self._metadata.segmentation_file).name

        actin_path = self._metadata.get("annotation_file_actin")
        self._results.annotation_file_actin = Path(actin_path).name if actin_path else None

        self._results.cell_type = self._metadata.cell_type
        self._results.condition = self._metadata.condition
        self._results.adherence_time = self._metadata.adherence_time
        self._results.replicate_nr = self._metadata.replicate_nr
        self._results.pillar_step = self._metadata.pillar_step

    def _load_image(self):
        """
        Load the image stack, the segmentation(s) and the annotation(s), and crop the
        segmentations to the annotated regions.

        The nucleus and condensate segmentations are cropped to the nucleus annotation,
        the actin segmentation to the actin annotation. `crop_region` returns, with each
        cropped array, a dict with the "scale" (voxel size) and "translate" (position of
        the crop in the full image, in µm) needed to convert voxel indices of the crop to
        physical coordinates: physical = voxel * scale + translate.

        Returns
        -------
        channels : tuple of ndarray
            DAPI, HP1α, actin (None if the stack has only three channels) and pillar
            channels, each (z, y, x).
        scaling_factors : ndarray
            Voxel size (z, y, x) in µm, from the image metadata.
        annotation_nucleus, annotation_actin : ndarray
            Annotation polygons. `annotation_actin` is None if there is no actin annotation.
        shapes_layer_nucleus, shapes_layer_actin : napari.layers.Shapes
            The annotations as napari shapes layers (`shapes_layer_actin` may be None).
        cropped_cond, props_cond : ndarray, dict
            Cropped HP1α condensate mask (class 4) and its crop properties.
        cropped_nuc, props_nuc : ndarray, dict
            Cropped nucleus mask (classes 2–4) and its crop properties.
        cropped_actin_foci, props_actin_foci, cropped_actin_fibers, props_actin_fibers
            Cropped actin masks and their crop properties, or None if there is no actin
            annotation.

        Raises
        ------
        FileNotFoundError
            If the image file does not exist.
        """

        # Image stack: used for the voxel size, the pillar channel and the viewer
        image_file = self._data_dir / self._metadata.image_file
        if not image_file.exists():
            raise FileNotFoundError(
                f"Could not find '{image_file}'. Check if provided 'data_dir' and 'image_file' are correct."
            )

        image_stack = AICSImage(image_file)
        scale = image_stack.physical_pixel_sizes
        z_res, y_res, x_res = scale.Z, scale.Y, scale.X
        scaling_factors = np.array([z_res, y_res, x_res])

        self._results.physical_pixel_size_z = z_res
        self._results.physical_pixel_size_x = x_res
        self._results.physical_pixel_size_y = y_res
        self._results.pixel_size_um2 = y_res * x_res
        self._results.voxel_size_um3 = z_res * y_res * x_res

        # Channel order: DAPI, HP1α, (actin), pillars. Stacks without actin staining have
        # three channels.
        channel_DAPI = image_stack.data[0, 0, :, :, :]
        channel_HP1a = image_stack.data[0, 1, :, :, :]
        if image_stack.data.shape[1] >= 4:
            channel_actin = image_stack.data[0, 2, :, :, :]
            channel_pillar = image_stack.data[0, 3, :, :, :]
        else:
            channel_actin = None
            channel_pillar = image_stack.data[0, 2, :, :, :]

        # Ilastik segmentation of the HP1α channel. The nucleus is everything that is not
        # background (nucleoplasm, nucleoli and condensates).
        with h5py.File(self._metadata.segmentation_file, "r") as f:
            seg_map = np.squeeze(f["exported_data"][:])
        nucleus_mask = seg_map >= NUCLEOPLASM
        HP1a_mask = seg_map == HP1A_CONDENSATES

        image_layer_ilastik_nucleus = Image(data=nucleus_mask, scale=scaling_factors, name="Image_ilastik_nucleus")
        image_layer_ilastik_cond = Image(data=HP1a_mask, scale=scaling_factors, name="Image_ilastik_cond")

        # Ilastik segmentation of the actin channel (optional)
        image_layer_ilastik_actin_foci = None
        image_layer_ilastik_actin_fibers = None
        seg_file_actin = self._metadata.segmentation_file_actin
        if seg_file_actin:
            with h5py.File(seg_file_actin, "r") as f:
                seg_map_actin = np.squeeze(f["exported_data"][:])
            image_layer_ilastik_actin_foci = Image(
                data=seg_map_actin == ACTIN_FOCI, scale=scaling_factors, name="Image_ilastik_actin_foci"
            )
            image_layer_ilastik_actin_fibers = Image(
                data=seg_map_actin == ACTIN_FIBERS, scale=scaling_factors, name="Image_ilastik_actin_fibers"
            )

        # Nucleus annotation: polygon drawn around the nucleus in napari
        annotation_nucleus = np.load(self._metadata.annotation_file_nucleus)
        shapes_layer_nucleus = Shapes(data=annotation_nucleus, shape_type="polygon", name="Annotation_nucleus")

        # Actin annotation (optional): polygon drawn around the cell
        annotation_actin = None
        shapes_layer_actin = None
        cropped_actin_foci = cropped_actin_fibers = None
        props_actin_foci = props_actin_fibers = None

        actin_file = self._metadata.get("annotation_file_actin")
        if actin_file:
            annotation_actin = np.load(actin_file)
            shapes_layer_actin = Shapes(data=annotation_actin, shape_type="polygon", name="Annotation_actin")

            cropped_actin_foci, props_actin_foci, _ = crop_region(
                layer=image_layer_ilastik_actin_foci, shapes_layer=shapes_layer_actin
            )[0]
            cropped_actin_fibers, props_actin_fibers, _ = crop_region(
                layer=image_layer_ilastik_actin_fibers, shapes_layer=shapes_layer_actin
            )[0]

        # Crop the condensate and nucleus segmentations to the nucleus annotation
        cropped_cond, props_cond, _ = crop_region(layer=image_layer_ilastik_cond, shapes_layer=shapes_layer_nucleus)[0]
        cropped_nuc, props_nuc, _ = crop_region(layer=image_layer_ilastik_nucleus, shapes_layer=shapes_layer_nucleus)[0]

        return (
            (channel_DAPI, channel_HP1a, channel_actin, channel_pillar),
            scaling_factors,
            annotation_nucleus,
            annotation_actin,
            shapes_layer_nucleus,
            shapes_layer_actin,
            cropped_cond,
            props_cond,
            cropped_nuc,
            props_nuc,
            cropped_actin_foci,
            props_actin_foci,
            cropped_actin_fibers,
            props_actin_fibers,
        )

    # ------------------------------------------------------------------ segmentation

    def _segment_condensates(self, cropped_cond):
        """
        Segment the HP1α condensates in the cropped Ilastik condensate mask.

        The mask is smoothed with a Gaussian blur in the xy-plane only (σ_xy = 1 voxel,
        σ_z = 0, because the z-step is about twice the pixel size), thresholded with
        Otsu's method and split into objects by connected-component labelling.

        Returns
        -------
        labeled_HP1 : ndarray
            Label image of the condensates, with labels 1..N.
        """

        blurred_HP1a = cle.gaussian_blur(cropped_cond, None, sigma_x=1, sigma_y=1, sigma_z=0)
        binary_HP1 = cle.threshold_otsu(blurred_HP1a, None)
        labeled_HP1 = cle.exclude_small_labels(label(binary_HP1), None, self._min_condensate_voxels)
        # Relabel with scikit-image to get a NumPy label image
        return label(labeled_HP1)

    def _segment_nucleus(self, cropped_nuc):
        """
        Segment the nucleus in the cropped Ilastik nucleus mask.

        Same blur and Otsu threshold as for the condensates. The largest connected object
        is kept as the nucleus, so that fragments of neighbouring nuclei inside the
        annotation are removed.

        Returns
        -------
        labeled_nucleus : ndarray of uint16
            Binary image of the nucleus (1 = nucleus).

        Raises
        ------
        ValueError
            If no object is found.
        """

        blurred_nucleus = cle.gaussian_blur(cropped_nuc, None, sigma_x=1, sigma_y=1, sigma_z=0)
        binary_nucleus = cle.threshold_otsu(blurred_nucleus, None)
        full_labeled_nucleus = label(binary_nucleus)

        objects = regionprops(full_labeled_nucleus)
        if not objects:
            raise ValueError("No nucleus found in the segmentation.")
        largest = max(objects, key=lambda obj: obj.area)

        return (full_labeled_nucleus == largest.label).astype(np.uint16)

    def _segment_pillars(self, channel_pillar, scaling_factors, shapes_for_pillar):
        """
        Segment the pillars in the pillar channel of the image stack and crop them to the
        region of interest.

        The full channel is blurred in the xy-plane and thresholded with Otsu's method,
        or with `metadata["pillar_threshold"]` if it is set (used for images with weak
        pillar signal). The pillar labels are then cropped to `shapes_for_pillar`.

        Returns
        -------
        labeled_pillar : ndarray
            Pillar labels in the full image.
        labeled_pillar_cropped : ndarray
            Pillar labels cropped to the region of interest.
        props_pillars : dict
            Crop properties ("scale" and "translate") of `labeled_pillar_cropped`.
        """

        blurred_pillar = cle.gaussian_blur(channel_pillar, None, sigma_x=1, sigma_y=1, sigma_z=0)
        binary_pillar = cle.threshold_otsu(blurred_pillar, None)

        threshold = self._metadata.pillar_threshold
        if threshold is not None:
            binary_pillar = cle.greater_constant(blurred_pillar, None, float(threshold))

        labeled_pillar = label(binary_pillar)

        image_layer_pillar_labels = Image(data=labeled_pillar, scale=scaling_factors, name="Image_Pillar_labels")
        labeled_pillar_cropped, props_pillars, _ = crop_region(
            layer=image_layer_pillar_labels, shapes_layer=shapes_for_pillar
        )[0]

        return labeled_pillar, labeled_pillar_cropped, props_pillars

    # ------------------------------------------------------------------ measurements

    def _extract_condensate_properties(self, labeled_HP1):
        """
        Count the condensates and measure their volumes.

        The SimpleITK image has unit spacing, so `GetPhysicalSize` returns the number of
        voxels. Multiply by `voxel_size_um3` to get µm³.
        """

        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(sitk.GetImageFromArray(labeled_HP1.astype(int)))

        condensate_ids = shape_stats.GetLabels()
        print("Total number of condensates:", len(condensate_ids))

        self._results.condensate_ids = condensate_ids
        self._results.tot_num_condensates = len(condensate_ids)
        self._results.voxels_condensates = [shape_stats.GetPhysicalSize(i) for i in condensate_ids]

    def _extract_nucleus_properties(self, labeled_nucleus):
        """
        Measure the volume and sphericity of the nucleus.

        As for the condensates, the volume is a number of voxels. Sphericity is computed by
        pyradiomics on the voxel grid with unit spacing.
        """

        labeled_nucleus_sitk = sitk.GetImageFromArray(labeled_nucleus.astype(int))
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(labeled_nucleus_sitk)
        nucleus_id = shape_stats.GetLabels()[0]

        shape_features = shape.RadiomicsShape(labeled_nucleus_sitk, labeled_nucleus_sitk)

        self._results.nucleus_id = nucleus_id
        self._results.voxels_nucleus = shape_stats.GetPhysicalSize(nucleus_id)
        self._results.sphericity = shape_features.getSphericityFeatureValue()

    def _extract_nucleus_axes(self, labeled_nucleus, props_nuc):
        """
        Measure the axis lengths and flatness of the nucleus, and save its coordinates.

        The axis lengths are computed by `utils.axis_lengths`: a 2D PCA in the xy-plane
        gives the major and intermediate axes, and the minor axis is the extent along z.
        For the nucleus each extent is measured between the 1st and 99th percentile of the
        voxel positions, which makes it robust to single protruding voxels.
        Flatness = 1 − √(minor / major).

        Returns
        -------
        nucleus_coord : ndarray
            Voxel coordinates (z, y, x) of the nucleus in the cropped image.
        physical_coord_nucleus : ndarray
            Physical coordinates of the nucleus. Note that `axis_lengths` swaps the z and
            x columns of this array in place; the saved file has the (z, y, x) order.
        """

        nucleus_coord = np.array(find_coordinates(labeled_nucleus))
        physical_coord_nucleus = nucleus_coord * props_nuc["scale"] + props_nuc["translate"]

        # Saved before axis_lengths reorders the columns
        self._save_coordinates("nuclei", nucleus_coord, physical_coord_nucleus)

        _, x_length_nuc, y_length_nuc, z_length_nuc = axis_lengths(physical_coord_nucleus, perc=1)

        self._results.major_axis_nucleus = x_length_nuc
        self._results.intermediate_axis_nucleus = y_length_nuc
        self._results.minor_axis_nucleus = z_length_nuc
        self._results.flatness = 1 - np.sqrt(z_length_nuc / x_length_nuc)

        return nucleus_coord, physical_coord_nucleus

    def _extract_condensate_axes(self, labeled_HP1, scaling_factors):
        """
        Measure the axis lengths of each condensate, as for the nucleus but using the full
        extent (0th–100th percentile), because condensates are only a few voxels wide.
        """

        major, intermediate, minor = [], [], []
        for coords in find_coordinates(labeled_HP1, one_object=False):
            physical_coords = np.array(scale_coordinates(coords, scaling_factors))
            _, x_length, y_length, z_length = axis_lengths(physical_coords, perc=0)
            major.append(x_length)
            intermediate.append(y_length)
            minor.append(z_length)

        self._results.major_axis_HP1 = major
        self._results.intermediate_axis_HP1 = intermediate
        self._results.minor_axis_HP1 = minor

    def _extract_centroids(self, labeled_HP1, labeled_nucleus, props_nuc, props_cond):
        """
        Compute the centroids of the nucleus and of each condensate.

        Returns
        -------
        centroid_nucleus : ndarray, shape (1, 3)
            Centroid of the nucleus (voxels, cropped image).
        physical_centroid_nucleus : ndarray, shape (1, 3)
            Centroid of the nucleus in µm.
        centroid_conds_arr : ndarray, shape (N, 3)
            Centroids of the condensates (voxels, cropped image).
        physical_centroid_conds_arr : ndarray, shape (N, 3)
            Centroids of the condensates in µm.
        """

        centroid_nucleus = np.array([obj.centroid for obj in regionprops(labeled_nucleus)])
        physical_centroid_nucleus = centroid_nucleus * props_nuc["scale"] + props_nuc["translate"]

        centroid_conds_arr = np.array([obj.centroid for obj in regionprops(labeled_HP1)])
        physical_centroid_conds_arr = centroid_conds_arr * props_cond["scale"] + props_cond["translate"]

        return centroid_nucleus, physical_centroid_nucleus, centroid_conds_arr, physical_centroid_conds_arr

    def _define_nucleus_periphery(self, nucleus_coord, centroid_nucleus, labeled_nucleus, props_nuc):
        """
        Define the nuclear periphery as a thin outer shell of the nucleus.

        Every nucleus voxel is moved 5% of the way towards the centroid
        (`lmbda_dist_out_from_center` with λ = 0.95), which gives a slightly shrunken copy
        of the nucleus. The periphery is the nucleus minus this inner object.

        Returns
        -------
        periphery_coord_arr : ndarray
            Voxel coordinates of the periphery (cropped image).
        physical_periphery_arr : ndarray
            Physical coordinates of the periphery in µm.
        """

        labeled_inner_nuc, _ = lmbda_dist_out_from_center(0.95, nucleus_coord, centroid_nucleus, labeled_nucleus)
        periphery = cle.binary_subtract(labeled_nucleus, labeled_inner_nuc)

        periphery_coord_arr = np.array(find_coordinates(periphery))
        physical_periphery_arr = periphery_coord_arr * props_nuc["scale"] + props_nuc["translate"]

        return periphery_coord_arr, physical_periphery_arr

    def _radial_distances(self, physical_centroid_nucleus, physical_centroid_conds_arr):
        """
        Compute the Euclidean distance (µm) from the nucleus centroid to the centroid of
        each condensate.
        """

        self._results.r_dist_condensates = [
            LA.norm([centroid] - physical_centroid_nucleus) for centroid in physical_centroid_conds_arr
        ]

    def _min_peripheral_distances(self, physical_centroid_conds_arr, physical_periphery_arr):
        """
        Compute the distance (µm) from the centroid of each condensate to the closest
        point of the nuclear periphery.

        Returns
        -------
        coord_edge_p_cond : ndarray
            The closest periphery point for each condensate (shown in the viewer).
        """

        distances = cdist(physical_centroid_conds_arr, physical_periphery_arr)

        self._results.p_minimal_distance_condensates = np.min(distances, axis=1).tolist()

        return physical_periphery_arr[np.argmin(distances, axis=1)]

    # ------------------------------------------------------------------ coordinate files

    def _save_coordinates(self, object_type, voxel_coords, physical_coords, file_label=""):
        """
        Save voxel and physical coordinates of one type of object.
        """

        # Image name without the .ome.tif extension
        image_name = Path(self._metadata.image_file).with_suffix("").with_suffix("").name
        m = re.search(r"nuc(\d+)", Path(self._meta_file).stem)
        nuc_nr = m.group(1) if m else "unknown"
        prefix = f"{image_name}_nuc{nuc_nr}" + (f"_{file_label}" if file_label else "")

        for kind, coords in (("voxel", voxel_coords), ("physical", physical_coords)):
            out_dir = Path(self._coordinate_files) / f"{kind}_coordinates" / object_type
            out_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(coords, list):
                np.savez(out_dir / f"{prefix}_{kind}_coords.npz", *coords)
            else:
                np.save(out_dir / f"{prefix}_{kind}_coords.npy", coords)

    def _extract_condensate_coordinates(self, labeled_HP1, props_cond):
        """Save the voxel and physical coordinates of each condensate."""

        coords_voxel = find_coordinates(labeled_HP1, one_object=False)
        coords_physical = [c * props_cond["scale"] + props_cond["translate"] for c in coords_voxel]
        self._save_coordinates("HP1a_condensates", coords_voxel, coords_physical, "condensate")

    def _extract_actin_coordinates(
        self, cropped_actin_foci, props_actin_foci, cropped_actin_fibers, props_actin_fibers
    ):
        """
        Save the voxel and physical coordinates of actin foci and fibers (used by the
        actin-nucleus localization analysis). The cropped Ilastik masks are labelled
        directly, without blurring. Nothing is saved if there is no actin annotation.
        """

        for cropped, props, name in (
            (cropped_actin_foci, props_actin_foci, "actin_foci"),
            (cropped_actin_fibers, props_actin_fibers, "actin_fibers"),
        ):
            if cropped is None or props is None:
                continue
            coords_voxel = find_coordinates(label(cropped), one_object=False)
            coords_physical = [c * props["scale"] + props["translate"] for c in coords_voxel]
            self._save_coordinates(name, coords_voxel, coords_physical, name)

    def _extract_pillar_coordinates(self, labeled_pillar_cropped, props_pillars):
        """
        Save the voxel and physical coordinates of each pillar (used by the nuclear
        surface geometry analysis).

        Objects with fewer than 10% of the voxels of the largest object are discarded as
        noise or partially cropped pillars.

        Returns
        -------
        filtered_pillars_voxel_v, filtered_pillars_physical_v : ndarray
            Coordinates of all kept pillars stacked into one array (for the viewer).
        """

        pillar_coord = find_coordinates(labeled_pillar_cropped, one_object=False)
        pillar_coord_physical = [c * props_pillars["scale"] + props_pillars["translate"] for c in pillar_coord]

        max_points = max(c.shape[0] for c in pillar_coord_physical)
        threshold = 0.1 * max_points
        print(f"Keeping pillars with ≥ {threshold:.0f} points (of max {max_points})")

        filtered_pillars_voxel = [c for c in pillar_coord if c.shape[0] >= threshold]
        filtered_pillars_physical = [c for c in pillar_coord_physical if c.shape[0] >= threshold]

        self._save_coordinates("pillars", filtered_pillars_voxel, filtered_pillars_physical, "pillar")

        return np.vstack(filtered_pillars_voxel), np.vstack(filtered_pillars_physical)

    # ------------------------------------------------------------------ viewer

    def _viewer(
        self,
        channel_DAPI,
        channel_HP1a,
        channel_actin,
        channel_pillar,
        scaling_factors,
        annotation_nucleus,
        annotation_actin,
        cropped_nuc,
        props_nuc,
        cropped_cond,
        props_cond,
        cropped_actin_foci,
        props_actin_foci,
        cropped_actin_fibers,
        props_actin_fibers,
        labeled_nucleus,
        labeled_HP1,
        labeled_pillar,
        labeled_pillar_cropped,
        props_pillars,
        filtered_pillars_voxel_v,
        filtered_pillars_physical_v,
        nucleus_coord,
        physical_coord_nucleus,
        periphery_coord_arr,
        physical_centroid_nucleus,
        physical_centroid_conds_arr,
        centroid_conds_arr,
        coord_edge_p_cond,
    ):
        """
        Show the images, segmentations and measured points in a 3D napari viewer, for
        visual inspection.

        Layers of the cropped region are placed with the crop's "scale" and "translate".
        Point layers in µm need no transform.
        """

        viewer = napari.Viewer()
        image_kwargs = {"blending": "additive", "scale": scaling_factors, "rendering": "mip"}

        # Image channels
        viewer.add_image(
            np.squeeze(channel_HP1a), colormap="green", name="HP1a", contrast_limits=(0, 400), **image_kwargs
        )
        viewer.add_image(channel_DAPI, colormap="blue", name="DAPI", visible=False, **image_kwargs)
        if channel_actin is not None:
            viewer.add_image(channel_actin, colormap="red", name="Actin", **image_kwargs)
        viewer.add_image(channel_pillar, colormap="yellow", name="Pillar", visible=False, **image_kwargs)

        # Annotations
        viewer.add_shapes(
            annotation_nucleus,
            shape_type="polygon",
            blending="additive",
            scale=scaling_factors,
            name="Shape annotation nucleus",
            visible=False,
        )
        if annotation_actin is not None:
            viewer.add_shapes(
                annotation_actin,
                shape_type="polygon",
                blending="additive",
                scale=scaling_factors,
                name="Shape annotation actin",
                visible=False,
            )

        # Cropped Ilastik masks and segmentations
        nuc_place = {"scale": props_nuc["scale"], "translate": props_nuc["translate"]}
        cond_place = {"scale": props_cond["scale"], "translate": props_cond["translate"]}

        viewer.add_image(
            cropped_nuc, blending="additive", rendering="mip", name="cropped_data_nucleus", visible=False, **nuc_place
        )
        viewer.add_labels(labeled_nucleus, blending="additive", name="labeled_nucleus", **nuc_place)
        viewer.add_image(
            cropped_cond, blending="additive", rendering="mip", name="cropped_data_HP1a", visible=False, **cond_place
        )
        viewer.add_labels(labeled_HP1, blending="additive", name="labeled_HP1", **cond_place)

        for cropped, props, name in (
            (cropped_actin_foci, props_actin_foci, "labeled_actin_foci"),
            (cropped_actin_fibers, props_actin_fibers, "labeled_actin_fibers"),
        ):
            if cropped is not None:
                viewer.add_labels(
                    label(cropped),
                    blending="additive",
                    scale=props["scale"],
                    translate=props["translate"],
                    name=name,
                    visible=False,
                )

        # Pillars
        if labeled_pillar is not None:
            viewer.add_labels(
                labeled_pillar, blending="additive", scale=scaling_factors, name="labeled_pillar", visible=False
            )
        if labeled_pillar_cropped is not None:
            viewer.add_labels(
                labeled_pillar_cropped,
                blending="additive",
                scale=props_pillars["scale"],
                translate=props_pillars["translate"],
                name="labeled_pillar_cropped",
                visible=False,
            )
        if filtered_pillars_voxel_v is not None and filtered_pillars_voxel_v.size > 0:
            viewer.add_points(
                filtered_pillars_voxel_v,
                blending="translucent",
                face_color="orange",
                size=1,
                scale=props_pillars["scale"],
                translate=props_pillars["translate"],
                name="filtered_pillars_voxel",
            )
        if filtered_pillars_physical_v is not None and filtered_pillars_physical_v.size > 0:
            viewer.add_points(
                filtered_pillars_physical_v,
                blending="translucent",
                face_color="orange",
                size=1,
                name="filtered_pillars_physical",
                visible=False,
            )

        # Points used in the measurements
        point_kwargs = {"blending": "translucent", "size": 1, "visible": False}
        viewer.add_points(nucleus_coord, face_color="orange", name="nucleus_coords", **nuc_place, **point_kwargs)
        # physical_coord_nucleus has been reordered to (x, y, z) by axis_lengths
        viewer.add_points(physical_coord_nucleus, face_color="orange", name="nucleus_coords_physical", **point_kwargs)
        viewer.add_points(periphery_coord_arr, face_color="orange", name="periphery", **nuc_place, **point_kwargs)
        viewer.add_points(physical_centroid_nucleus, face_color="yellow", name="centroid_nucleus", **point_kwargs)
        viewer.add_points(physical_centroid_conds_arr, face_color="red", name="centroid_condensates", **point_kwargs)
        viewer.add_points(
            centroid_conds_arr,
            blending="translucent",
            face_color="red",
            size=10,
            visible=False,
            name="centroid_condensates_voxel",
            **cond_place,
        )
        viewer.add_points(
            coord_edge_p_cond, face_color="magenta", name="peripheral_point_for_condensates", **point_kwargs
        )

        viewer.dims.ndisplay = 3
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "um"

        return viewer
