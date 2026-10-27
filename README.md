# Lodox_pipeline

Bone suppression and image processing pipeline for the EEE4022S thesis project "The development of a low-cost X-ray phantom for bone suppression and radiography system calibration" (M Saeed Solomon, SLMMOG032, UCT).

The whole pipeline runs from `Bone_suppression_pipeline.ipynb`, and the functions it uses are in `Modules/`. The notebook works through these steps:

1. Load the low and high kVp scans (60 and 120 kVp at 80 mA).
2. Register the high kVp scan onto the low kVp scan.
3. Find the ROIs.
4. Calibrate with the PLA step wedge.
5. Suppress the bone (weighted log subtraction).
6. Suppress the noise (Noise2Noise).
7. Validate the result.

## Data (not included)

The DICOM (`.dcm`) scans are too large for GitHub, so they are not in this repository. To run the notebook:

1. Set the `folder` of each session in `SESSIONS` (the Phase 1 cell) to the folder that holds your scans.
2. Set `prefix`, `low`, `high`, `repeat_low` and `repeat_high` to match your file names.

Each session needs a low/high kVp pair and a repeat of that pair. Noise2Noise trains on the repeat as its second, independent noisy image.

The pipeline expects unprocessed Lodox Statscan images (`ORIGINAL\PRIMARY` log images), not processed or exported ones.

## Masks are specific to this experiment

The ROI masks were built for the phantoms, ice trays and step wedge used in this project. They are not generic, so other scans will probably need changes:

- `Modules/Regions.py`: fixed boxes (`SESSION_BOXES_MM`, in mm from the top-left of the image) give where each object lay on the scanner in the 21 and 25 October sessions. A different layout needs new boxes.
- `Modules/Ice_tray_mask.py`: assumes two upright 2 × 6 ice trays.
- `Modules/Wedge_mask.py`: assumes a five-step 1–5 cm PLA wedge.
- `Modules/Phantom_mask.py`: assumes the bone/muscle/fat layering of these phantoms, and the CaCO₃ sediment bone of prototype 3.

The subtraction weight and effective energies are measured for this Lodox unit and these presets. Recalibrate them on another system rather than reusing the numbers.

## Requirements

Tested with Python 3.14 and the following packages:

- numpy, scipy, pandas
- scikit-image, matplotlib, Pillow
- pydicom
- torch (CPU is enough)
- xraydb

## Outputs

- Results, figures and the trained Noise2Noise model (`noise2noise.pt`) are written to `outputs/<scan IDs>/`. This folder is git-ignored. A saved model is reused on later runs.
- On the first run, the validation step downloads three humerus radiographs from a public Kaggle dataset (CC BY 4.0) into `../Open Source Images/`.
