"""
Helper functions used by `mechcell.process`.

Coordinates are arrays with one row per voxel and the columns in the (z, y, x) order of
the image stack, unless stated otherwise.
"""

import numpy as np
import pyclesperanto_prototype as cle
from skimage.measure import label
from sklearn.decomposition import PCA


class dotdict(dict):
    """
    Dictionary whose keys can also be read as attributes: `d.key` is `d.get("key")`.

    A missing key gives None instead of raising an error, which is how optional metadata
    entries (for example `segmentation_file_actin`) are handled.
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def find_coordinates(labeled_object, one_object=True):
    """
    Return the voxel coordinates of the objects in a label image.

    Parameters
    ----------
    labeled_object : ndarray
        3D label image (0 = background).
    one_object : bool, optional
        If True, return only the coordinates of the object with the lowest label.
        If False, return the coordinates of every object.

    Returns
    -------
    ndarray or list of ndarray
        Array of shape (n_voxels, 3) with (z, y, x) indices, or a list of such arrays,
        one per object in increasing label order.
    """

    coords_per_object = []
    labels = np.unique(labeled_object)
    labels = labels[labels != 0]  # skip background

    for lbl in labels:
        z, y, x = np.where(labeled_object == lbl)
        coords_per_object.append(np.column_stack((z, y, x)))

    if one_object:
        return coords_per_object[0]
    return coords_per_object


def scale_coordinates(coords, scaling_factors):
    """
    Convert voxel indices to µm by multiplying each axis by its voxel size.

    Parameters
    ----------
    coords : ndarray
        Voxel coordinates, shape (n, 3), in (z, y, x) order.
    scaling_factors : sequence of float
        Voxel size (z, y, x) in µm.

    Returns
    -------
    list of tuple
        Physical coordinates (z, y, x) in µm, one tuple per voxel.
    """

    return [(scaling_factors[0] * z, scaling_factors[1] * y, scaling_factors[2] * x) for z, y, x in coords]


def axis_lengths(physical_coord, perc):
    """
    Measure the major, intermediate and minor axis lengths of an object.

    1. The z and x columns of `physical_coord` are swapped, so the columns become
       (x, y, z). **This modifies the input array in place.**
    2. The object is centred on the midpoint of its bounding box.
    3. A 2D PCA in the xy-plane rotates the object so that its longest in-plane direction
       lies along the first axis. The z axis is not rotated.
    4. Each axis length is the extent of the rotated coordinates along that axis, measured
       between the `perc` and `100 - perc` percentiles. `perc > 0` makes the lengths robust
       to single protruding voxels.

    The major and intermediate axes are therefore the in-plane extents along and across
    the main direction of the object, and the minor axis is its height along z.

    Parameters
    ----------
    physical_coord : ndarray
        Physical coordinates of the object in µm, shape (n, 3), in (z, y, x) order.
        Reordered to (x, y, z) in place.
    perc : float
        Percentile (0–50) cut off at both ends of each axis. 1 for nuclei, 0 for
        condensates.

    Returns
    -------
    rotocentered_physical_coord : ndarray
        Centred and rotated coordinates, columns (major, intermediate, z).
    x_length_adj : float
        Major axis length (µm).
    y_length_adj : float
        Intermediate axis length (µm).
    z_length_adj : float
        Minor axis length (µm), the extent along z.
    """

    # Reorder the columns from (z, y, x) to (x, y, z)
    physical_coord[:, [0, 2]] = physical_coord[:, [2, 0]]

    # Centre on the midpoint of the bounding box
    T_x = (np.min(physical_coord[:, 0]) + np.max(physical_coord[:, 0])) / 2
    T_y = (np.min(physical_coord[:, 1]) + np.max(physical_coord[:, 1])) / 2
    T_z = (np.min(physical_coord[:, 2]) + np.max(physical_coord[:, 2])) / 2
    centered_physical_coord = physical_coord - np.array([T_x, T_y, T_z])

    # Rotate the xy-plane onto its principal axes
    pca = PCA(n_components=2)
    pca.fit(centered_physical_coord[:, 0:2])
    rotocentered_physical_coord = centered_physical_coord.copy()
    rotocentered_physical_coord[:, 0:2] = np.matmul(rotocentered_physical_coord[:, 0:2], np.transpose(pca.components_))

    # Extent along each axis between the perc and 100 - perc percentiles
    x_length_adj = np.percentile(rotocentered_physical_coord[:, 0], 100 - perc) - np.percentile(
        rotocentered_physical_coord[:, 0], perc
    )
    y_length_adj = np.percentile(rotocentered_physical_coord[:, 1], 100 - perc) - np.percentile(
        rotocentered_physical_coord[:, 1], perc
    )
    z_length_adj = np.percentile(rotocentered_physical_coord[:, 2], 100 - perc) - np.percentile(
        rotocentered_physical_coord[:, 2], perc
    )

    return rotocentered_physical_coord, x_length_adj, y_length_adj, z_length_adj


def lmbda_dist_out_from_center(lmbda, coord_nuc, centroid_nuc, labeled_nucleus):
    """
    Make a copy of the nucleus by moving every voxel towards the centroid.

    Each voxel p becomes `lmbda * p + (1 - lmbda) * centroid`, so `lmbda = 1` gives the
    nucleus itself and smaller values give a smaller copy around the centroid. The moved
    points are rounded down to voxel indices and labelled, and only the part that lies
    inside `labeled_nucleus` is kept. `mechcell.process` uses `lmbda = 0.95` and defines
    the nuclear periphery as the nucleus minus this copy.

    Parameters
    ----------
    lmbda : float
        Scale factor between 0 and 1.
    coord_nuc : ndarray
        Voxel coordinates (z, y, x) of the nucleus, shape (n, 3).
    centroid_nuc : ndarray
        Voxel coordinates (z, y, x) of the nucleus centroid, shape (1, 3).
    labeled_nucleus : ndarray
        Binary 3D image of the nucleus.

    Returns
    -------
    labeled : ndarray
        Binary image of the shrunken nucleus.
    coord_layer : ndarray
        Voxel coordinates (z, y, x) of the shrunken nucleus.
    """

    # Move every voxel towards the centroid
    new_points = lmbda * coord_nuc + (1 - lmbda) * centroid_nuc

    # Draw the moved points into a binary image large enough to hold them
    image_shape = tuple(int(np.max(new_points[:, axis]) + 1) for axis in range(3))
    binary_image = np.zeros(image_shape, dtype=bool)
    new_points_int = new_points.astype(int)
    binary_image[new_points_int[:, 0], new_points_int[:, 1], new_points_int[:, 2]] = True

    # Keep only the part that overlaps the nucleus
    labeled = cle.binary_and(label(binary_image), labeled_nucleus)
    coord_layer = find_coordinates(labeled, one_object=True)

    return labeled, coord_layer
