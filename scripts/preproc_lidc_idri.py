"""LIDC-IDRI preprocessing script.

Converts the original LIDC-IDRI DICOM studies into NIfTI files and produces
one normalized ``processed.nii.gz`` volume per case.

Pipeline summary:

1. Convert each case directory from DICOM to NIfTI with ``dicom2nifti``.
2. Resample to isotropic 1 mm spacing.
3. Center crop to ``256 × 256 × 256``.
4. Clip HU values, suppress upper-tail outliers, and normalize to ``[0, 1]``.
5. Save the final volume as ``processed.nii.gz``.

Adapted from the preprocessing utility in the WDM repository
(``pfriedri/wdm-3d``) which documents the same ``preproc_lidc-idri.py``
workflow for DICOM-sourced LIDC-IDRI data.
"""

import argparse
import os
import shutil
import dicom2nifti
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def preprocess_nifti(input_path, output_path):
    """Convert one intermediate NIfTI scan into the final normalized training volume.

    Parameters
    ----------
    input_path : str
        Path to the intermediate NIfTI file produced by ``dicom2nifti``.
    output_path : str
        Destination path for the normalized ``processed.nii.gz`` volume.
    """
    # Load the Nifti image
    print('Process image: {}'.format(input_path))
    img = nib.load(input_path)

    # Get the current voxel sizes
    voxel_sizes = img.header.get_zooms()

    # Calculate the target voxel size (1mm x 1mm x 1mm)
    target_voxel_size = (1.0, 1.0, 1.0)

    # Calculate the resampling factor
    zoom_factors = [current / target for target, current in zip(target_voxel_size, voxel_sizes)]

    # Resample the image
    print("[1] Resample the image ...")
    resampled_data = zoom(img.get_fdata(), zoom_factors, order=3, mode='nearest')

    print("[2] Center crop the image ...")
    crop_size = (256, 256, 256)
    depth, height, width = resampled_data.shape

    d_start = (depth - crop_size[0]) // 2
    h_start = (height - crop_size[1]) // 2
    w_start = (width - crop_size[2]) // 2
    cropped_arr = resampled_data[d_start:d_start + crop_size[0], h_start:h_start + crop_size[1], w_start:w_start + crop_size[2]]

    print("[3] Clip all values below -1000 ...")
    cropped_arr[cropped_arr < -1000] = -1000

    print("[4] Clip the upper quantile (0.999) to remove outliers ...")
    out_clipped = np.clip(cropped_arr, -1000, np.quantile(cropped_arr, 0.999))

    print("[5] Normalize the image ...")
    out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))

    assert out_normalized.shape == (256, 256, 256), "The output shape should be (320,320,320)"

    print("[6] FINAL REPORT: Min value: {}, Max value: {}, Shape: {}".format(out_normalized.min(),
                                                                             out_normalized.max(),
                                                                             out_normalized.shape))
    print("-------------------------------------------------------------------------------")
    # Save the resampled image
    resampled_img = nib.Nifti1Image(out_normalized, np.eye(4))
    nib.save(resampled_img, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert original LIDC-IDRI DICOM studies into normalized "
            "`processed.nii.gz` volumes."
        )
    )
    parser.add_argument('--dicom_dir', type=str, required=True,
                        help='Directory containing the original dicom data')
    parser.add_argument('--nifti_dir', type=str, required=True,
                        help='Directory to store the processed nifti files')
    parser.add_argument('--delete_unprocessed', type=eval, default=True,
                        help='Set true to delete the unprocessed nifti files')
    args = parser.parse_args()

    # Convert DICOM to nifti
    for item in os.listdir(args.dicom_dir):
        item_path = os.path.join(args.dicom_dir, item)
        # Skip if not a directory
        if not os.path.isdir(item_path):
            print(f'Skipping non-directory item: {item}')
            continue
            
        print(f'Converting {item} to nifti')
        output_dir = os.path.join(args.nifti_dir, item)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            dicom2nifti.convert_directory(item_path, output_dir)
            # Only remove the directory if conversion was successful
            shutil.rmtree(item_path)
        except Exception as e:
            print(f"Error converting {item}: {str(e)}")
            continue

    # Preprocess nifti files
    for root, dirs, files in os.walk(args.nifti_dir):
        for file in files:
            try:
                preprocess_nifti(os.path.join(root, file), os.path.join(root, 'processed.nii.gz'))
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue

    # Delete unprocessed nifti files
    if args.delete_unprocessed:
        for root, dirs, files in os.walk(args.nifti_dir):
            for file in files:
                if file != 'processed.nii.gz':
                    try:
                        os.remove(os.path.join(root, file))
                    except Exception as e:
                        print(f"Error deleting file {file}: {str(e)}")
                        continue
