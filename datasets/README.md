# Dataset Guide

This repository uses two 3D CT datasets:

- `LIDC-IDRI`
- `RAD-ChestCT`

This guide explains how to preprocess them and generate the split files used by
training.

## LIDC-IDRI

1. Download the LIDC-IDRI dataset from
   [The Cancer Imaging Archive (TCIA)](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI).
2. Place the original DICOM data in `data/LIDC-IDRI/`.
3. Convert the original DICOM studies into normalized NIfTI volumes:

```bash
python scripts/preproc_lidc-idri.py \
  --dicom_dir data/LIDC-IDRI \
  --nifti_dir data/LIDC-IDRI
```

This script converts the original LIDC-IDRI DICOM case folders into NIfTI and
then writes one final preprocessed volume per case as:

```text
data/LIDC-IDRI/LIDC-IDRI-XXXX/processed.nii.gz
```

The LIDC preprocessing utility is adapted from the workflow documented in the
[WDM repository](https://github.com/pfriedri/wdm-3d).

4. Generate the split files:

```bash
python scripts/split_train_val.py data/LIDC-IDRI \
  --dataset lidc \
  --output-dir lidc_data \
  --val-size 200
```

This creates:
- `lidc_data/train.txt`
- `lidc_data/val.txt`

## RAD-ChestCT

1. Download RAD-ChestCT from the official source
   [Zenodo](https://zenodo.org/records/6406114#.Ytl6OXbMLAQ).
2. Place the raw `.npz` files in `data/RAD-ChestCT/`.
3. Preprocess the raw volumes:

```bash
python scripts/preprocess_rad_chestct.py \
  --source_dir data/RAD-ChestCT \
  --target_dir data/rad_chestct_preprocessed \
  --target-shape 256 256 256 \
  --recursive
```

The preprocessing pipeline:
- clips HU values to `[-1000, 1000]`
- center crops to `256 x 256 x 256`
- downsamples afterward if a smaller target shape is requested
- saves preprocessed `.npz` files with `ct` and `volume` keys

4. Generate the split files:

```bash
python scripts/split_train_val.py data/rad_chestct_preprocessed \
  --dataset rad \
  --output-dir rad_data \
  --val-size 200
```

This creates:
- `rad_data/train.txt`
- `rad_data/val.txt`
