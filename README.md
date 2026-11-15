# Supplementary code for the paper: Using Nanopillar Arrays to Induce Nuclear Deformation and Reveal the Effects of Indentation on HP1-α Condensation

This repository contains the image analysis code for the paper

> Hovet, O.\*, Årbogen, S. B. S.\*, Raffo, A., Progida, C., Sikorski, P., Paulsen, J.\
> \*Equal contribution.\
> Using Nanopillar Arrays to Induce Nuclear Deformation and Reveal the Effects of Indentation on HP1-α Condensation.

DOI will be made available upon publication. 
> <!-- TODO: journal, year and DOI -->

## Abstract

Cells continuously sense and respond to mechanical cues from their environment, and the nucleus is increasingly recognized as a central mechanosensitive organelle. Here we use nanopillar arrays as a controlled mechanical environment that imposes localized nuclear deformation through cell adhesion rather than externally applied force. Combining electron-beam-fabricated fluorescent nanopillars of two heights and three pitches with confocal imaging and a dedicated 3D geometry-modelling pipeline, we quantify nuclear indentation and the response of HP1α heterochromatin condensates in HeLa cells after 6 and 24 hour adhesion on nanopillars. We find that indentation depth depends strongly on pillar height and pitch and evolves over time, and that nanopillar-induced deformation modestly reshapes the size, number and spatial distribution of HP1α condensates.

## Software

The Python environment is managed with [uv](https://docs.astral.sh/uv/). All direct
dependencies are pinned in `pyproject.toml`, and `uv.lock` records the exact versions of
every package.

```bash
git clone https://github.com/odasho/mechano-pillars-HP1alpha.git
cd mechano-pillars-HP1alpha
uv sync --extra notebook --extra viewer
uv run jupyter lab
```

Our `mechcell` Python package contains the procedures for:

* segmenting nuclei, HP1α condensates and pillars from the Ilastik segmentations. Pixel classification was done with [ilastik](https://www.ilastik.org/download) 1.4.0rc8.
* extracting shape descriptors of the nuclei and condensates
* saving the voxel and physical coordinates of nuclei, condensates, actin and pillars


## Repository structure

* `src/mechcell` contains the `mechcell` package (`process.py`, `utils.py`).
* `analysis` contains the notebooks:
    * `process_image/process_image_with_viewer.ipynb`: runs `mechcell.ProcessImage` on selected nuclei, with an optional napari viewer. This notebook was used to process all images and to generate the tables and coordinate files.
    * `actin_nucleus_localization/`: distance between the nucleus centroid and the basal
      actin plane.
    * `intensity_analysis/`: HP1α intensity of each condensate and of the nucleoplasm.
* `data` contains the metadata JSON files of every nucleus, in the folder structure shown below.
* `ilastik` contains the trained Ilastik projects.

## Data structure

The JSON files are organised by replicate, adherence time and substrate. 

```
data/
├── REP1/                                         # REP1, REP2, REP3
│   ├── 6hours/                                   # 6hours, 24hours
│   │   ├── 1000nm/                               # pillar height
│   │   │   ├── 260108_1000nm_2umpitch_6h_hela_experiments_meta/
│   │   │   │   ├── 260108_1000nm_2umpitch_6h_stack1_nuc1.json   # one JSON per nucleus
│   │   │   │   └── ...
│   │   │   ├── 260108_1000nm_3umpitch_6h_hela_experiments_meta/
│   │   │   ├── 260108_1000nm_4umpitch_6h_hela_experiments_meta/
│   │   │   └── 260108_1000nm_control_6h_hela_experiments_meta/
│   │   ├── 1600nm/                               # same layout as 1000nm/
│   │   └── Imaging_dish/
│   │       └── 260218_6h_Imaging_dish_hela_experiments_meta/
│   └── 24hours/
├── REP2/
└── REP3/
```

## License

MIT
