"""Shared utility functions for dataset preprocessing, loading, and augmentation.

This module collects small helpers used across the repository for:
- logging and lightweight config containers
- loading `.npz` and `.nii.gz` volumes
- CT intensity transforms and denoising
- resizing, padding, cropping, and augmentation
- MONAI-based transform pipelines

The functions are grouped by purpose so the file is easier to scan and extend.
"""

import csv
import logging
import os

import cv2
import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    MapTransform,
    Orientationd,
    Resized,
    ToTensord,
)
from scipy.ndimage import gaussian_filter, zoom


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __init__(self, *args, **kwargs):
        """Initialize the formatter and detect terminal color support."""
        super().__init__(*args, **kwargs)
        self.use_colors = self._supports_color()

    def _supports_color(self):
        """Check if terminal supports colors."""
        try:
            import sys

            if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
                term = os.environ.get("TERM", "")
                if "color" in term or term in ["xterm", "xterm-256color", "screen", "tmux"]:
                    return True
            return False
        except Exception:
            return False

    def format(self, record):
        """Format a log record with ANSI colors when the terminal supports them."""
        timestamp = self.formatTime(record, self.datefmt)
        base_message = f"{timestamp} - {record.name} - {record.levelname} - {record.getMessage()}"

        if not self.use_colors:
            return base_message

        color = self.COLORS.get(record.levelname, "")
        colored_timestamp = f"\033[90m{timestamp}\033[0m"
        colored_level = f"{color}{self.BOLD}{record.levelname}\033[0m"
        logger_name = f"\033[94m{record.name}\033[0m"

        if record.levelname == "INFO":
            if (
                "Min:" in record.getMessage()
                or "Max:" in record.getMessage()
                or "Mean:" in record.getMessage()
                or "Std:" in record.getMessage()
            ):
                message = f"\033[96m{record.getMessage()}\033[0m"
            elif "expected [0,1] range" in record.getMessage():
                message = f"\033[92m✅ {record.getMessage()}\033[0m"
            elif "Final" in record.getMessage() and "size:" in record.getMessage():
                message = f"\033[93m📊 {record.getMessage()}\033[0m"
            else:
                message = f"{color}{record.getMessage()}\033[0m"
        elif record.levelname == "WARNING":
            if "outside [0,1] range" in record.getMessage():
                message = f"\033[91m⚠️  {record.getMessage()}\033[0m"
            else:
                message = f"{color}⚠️  {record.getMessage()}\033[0m"
        else:
            message = f"{color}{record.getMessage()}\033[0m"

        return f"{colored_timestamp} - {logger_name} - {colored_level} - {message}"


class DataConfig:
    """Minimal data configuration container used by older preprocessing utilities."""

    def __init__(self, data_path, task, roi_size):
        self.data_path = data_path
        self.task = task
        self.roi_size = roi_size


class Config:
    """Minimal wrapper config object holding a `DataConfig` instance."""

    def __init__(self, data_config):
        self.data = data_config


def save_idx_files_to_csv(file_names, indices, out_file):
    """Save selected file indices and names to a CSV file."""
    with open(out_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "File"])
        for idx in indices:
            writer.writerow([idx, file_names[idx]])


def resize_volume(volume, target_shape=(256, 256, 256)):
    """Resize a volume to target shape using trilinear interpolation."""
    factors = [float(t) / float(s) for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1, mode="constant", cval=-1000)


def center_crop_3d(volume: np.ndarray, target_shape: tuple = (380, 380, 380)) -> np.ndarray:
    """Center crop a 3D volume to target shape, or return `None` if too small."""
    current_shape = volume.shape
    if any(s < t for s, t in zip(current_shape, target_shape)):
        return None

    start_indices = [(s - t) // 2 for s, t in zip(current_shape, target_shape)]
    slices = tuple(slice(start, start + size) for start, size in zip(start_indices, target_shape))
    return volume[slices]


def window_transform(window_level=40, window_width=350):
    """Calculate window min and max values."""
    min_hu = window_level - window_width / 2
    max_hu = window_level + window_width / 2
    return min_hu, max_hu


def normalize(ctvol, lower_bound, upper_bound):
    """Clip an image tensor to bounds and normalize it to `[0, 1]`."""
    ctvol = torch.clamp(ctvol, lower_bound, upper_bound)
    ctvol = (ctvol - lower_bound) / (upper_bound - lower_bound)
    return ctvol


def transform_hu_to_density(volume, bone_attenuation_multiplier=1.0):
    """Transform Hounsfield units into a density-like representation."""
    volume = volume.to(torch.float32)
    air = torch.where(volume <= -800)
    soft_tissue = torch.where((-800 < volume) & (volume <= 350))
    bone = torch.where(350 < volume)

    density = torch.empty_like(volume)
    density[air] = volume[soft_tissue].min()
    density[soft_tissue] = volume[soft_tissue]
    density[bone] = volume[bone] * bone_attenuation_multiplier
    density -= density.min()
    density /= density.max()
    return density


def torchify_pixelnorm_pixelcenter(ctvol, pixel_bounds):
    """Normalize using `pixel_bounds` and shift by the ImageNet mean."""
    ctvol = torch.from_numpy(ctvol).type(torch.float)
    ctvol = normalize(ctvol, pixel_bounds[0], pixel_bounds[1])
    ctvol = ctvol - 0.449
    return ctvol


def apply_non_local_means_denoising(volume, patch_size=7, patch_distance=11, h=0.1):
    """Placeholder for non-local means denoising."""
    pass


def apply_gaussian_filter(volume, sigma=1.0):
    """Apply Gaussian smoothing to a 3D volume."""
    return gaussian_filter(volume, sigma=sigma)


def apply_bilateral_filter(volume, diameter=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filtering to a 3D volume slice by slice."""
    filtered_volume = np.zeros_like(volume)
    for i in range(volume.shape[0]):
        filtered_volume[i] = cv2.bilateralFilter(
            volume[i].astype(np.float32),
            d=diameter,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
        )
    return filtered_volume


def apply_denoising(volume, method="gaussian"):
    """Apply the selected denoising method to the volume."""
    if method == "none":
        print("Skipping denoising...")
        return volume
    if method == "non_local_means":
        return apply_non_local_means_denoising(volume)
    if method == "gaussian":
        return apply_gaussian_filter(volume, sigma=1.20)
    if method == "bilateral":
        return apply_bilateral_filter(volume)
    raise ValueError(f"Unknown denoising method: {method}")


def pad_slices(ctvol, max_slices):
    """Pad the slice axis until the volume reaches `max_slices`."""
    padding_needed = max_slices - ctvol.shape[0]
    assert padding_needed >= 0, "Image slices exceed max_slices by" + str(-1 * padding_needed)
    if padding_needed > 0:
        before_padding = int(padding_needed / 2.0)
        after_padding = padding_needed - before_padding
        ctvol = np.pad(
            ctvol,
            pad_width=((before_padding, after_padding), (0, 0), (0, 0)),
            mode="constant",
            constant_values=np.amin(ctvol),
        )
        assert ctvol.shape[0] == max_slices
    return ctvol


def pad_sides(ctvol, max_side_length):
    """Pad the spatial axes until both side lengths reach `max_side_length`."""
    needed_padding = 0
    for side in [1, 2]:
        padding_needed = max_side_length - ctvol.shape[side]
        if padding_needed > 0:
            before_padding = int(padding_needed / 2.0)
            after_padding = padding_needed - before_padding
            if side == 1:
                ctvol = np.pad(
                    ctvol,
                    pad_width=((0, 0), (before_padding, after_padding), (0, 0)),
                    mode="constant",
                    constant_values=np.amin(ctvol),
                )
                needed_padding += 1
            elif side == 2:
                ctvol = np.pad(
                    ctvol,
                    pad_width=((0, 0), (0, 0), (before_padding, after_padding)),
                    mode="constant",
                    constant_values=np.amin(ctvol),
                )
                needed_padding += 1
    if needed_padding == 2:
        assert ctvol.shape[1] == ctvol.shape[2] == max_side_length
    return ctvol


def pad_volume(ctvol, max_slices, max_side_length):
    """Pad a volume so it reaches the minimum requested 3D size."""
    if ctvol.shape[0] < max_slices:
        ctvol = pad_slices(ctvol, max_slices)
    if ctvol.shape[1] < max_side_length:
        ctvol = pad_sides(ctvol, max_side_length)
    return ctvol


def crop_specified_axis(ctvol, max_dim, axis):
    """Center crop a 3D volume along a single axis if it exceeds `max_dim`."""
    dim = ctvol.shape[axis]
    if dim > max_dim:
        amount_to_crop = dim - max_dim
        part_one = int(amount_to_crop / 2.0)
        part_two = dim - (amount_to_crop - part_one)
        if axis == 0:
            return ctvol[part_one:part_two, :, :]
        if axis == 1:
            return ctvol[:, part_one:part_two, :]
        if axis == 2:
            return ctvol[:, :, part_one:part_two]
    return ctvol


def single_crop_3d_fixed(ctvol, max_slices, max_side_length):
    """Center crop a volume to `[max_slices, max_side_length, max_side_length]`."""
    ctvol = crop_specified_axis(ctvol, max_slices, 0)
    ctvol = crop_specified_axis(ctvol, max_side_length, 1)
    ctvol = crop_specified_axis(ctvol, max_side_length, 2)
    return ctvol


def rand_pad(ctvol):
    """Introduce random padding on all six sides of a volume."""
    randpad = np.random.randint(low=0, high=15, size=(6))
    ctvol = np.pad(
        ctvol,
        pad_width=((randpad[0], randpad[1]), (randpad[2], randpad[3]), (randpad[4], randpad[5])),
        mode="constant",
        constant_values=np.amin(ctvol),
    )
    return ctvol


def rand_flip(ctvol):
    """Flip a volume along a random axis with 50% probability."""
    if np.random.randint(low=0, high=100) < 50:
        chosen_axis = np.random.randint(low=0, high=3)
        ctvol = np.flip(ctvol, axis=chosen_axis)
    return ctvol


def rand_rotate(ctvol):
    """Rotate a volume axially by a random multiple of 90 degrees with 50% probability."""
    if np.random.randint(low=0, high=100) < 50:
        chosen_k = np.random.randint(low=0, high=4)
        ctvol = np.rot90(ctvol, k=chosen_k, axes=(1, 2))
    return ctvol


def single_crop_3d_augment(ctvol, max_slices, max_side_length):
    """Crop a 3D volume with light random padding, flips, and rotations."""
    ctvol = rand_pad(ctvol)
    ctvol = single_crop_3d_fixed(ctvol, max_slices, max_side_length)
    ctvol = rand_flip(ctvol)
    ctvol = rand_rotate(ctvol)
    return np.ascontiguousarray(ctvol)


def sliceify(ctvol):
    """Reshape a grayscale stack into a 3-channel slice representation."""
    return np.reshape(ctvol, newshape=[int(ctvol.shape[0] / 3), 3, ctvol.shape[1], ctvol.shape[2]])


def reshape_3_channels(ctvol):
    """Convert a grayscale volume into a 3-channel representation when possible."""
    if ctvol.shape[0] % 3 == 0:
        ctvol = sliceify(ctvol)
    else:
        if (ctvol.shape[0] - 1) % 3 == 0:
            ctvol = sliceify(ctvol[:-1, :, :])
        elif (ctvol.shape[0] - 2) % 3 == 0:
            ctvol = sliceify(ctvol[:-2, :, :])
    return ctvol


class LoadNPZImage(MapTransform):
    """MONAI map transform that loads image data from `.npz` files."""

    def __init__(self, keys, image_key="ct"):
        super().__init__(keys)
        self.image_key = image_key

    def __call__(self, data):
        """Replace file paths with loaded numpy arrays for configured keys."""
        d = dict(data)
        for key in self.keys:
            if isinstance(d[key], str):
                file_path = d[key]
                if not file_path.endswith(".npz"):
                    raise ValueError(f"Unsupported file format: {file_path}")
                npz_data = np.load(file_path)
                d[key] = npz_data[self.image_key]
        return d


class LoadNIFTIImage(MapTransform):
    """MONAI map transform that loads image data from `.nii.gz` files."""

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        """Replace NIfTI file paths with loaded numpy arrays for configured keys."""
        d = dict(data)
        for key in self.keys:
            file_path = d[key]
            if not file_path.endswith(".nii.gz"):
                raise ValueError(f"Unsupported file format: {file_path}")
            nii_data = nib.load(file_path)
            d[key] = np.asarray(nii_data.dataobj)
        return d


def create_transformer(mode="training", augment=True, roi_size=(32, 32, 32), dtype=torch.float32):
    """Create a MONAI transform pipeline for loading and preprocessing CT volumes."""
    base_transforms = [
        LoadNPZImage(keys=["image"], image_key="ct"),
        Lambdad(keys=["image"], func=lambda x: apply_denoising(x, method="gaussian")),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Resized(keys=["image"], spatial_size=roi_size, mode="trilinear", align_corners=True),
    ]

    post_transforms = [
        ToTensord(keys=["image"]),
        Lambdad(
            keys=["image"],
            func=lambda x: transform_hu_to_density(
                torch.clamp(
                    x,
                    min=window_transform(window_level=40, window_width=350)[0],
                    max=window_transform(window_level=40, window_width=350)[1],
                ),
                bone_attenuation_multiplier=1.0,
            ),
        ),
    ]

    final_transforms = [
        Lambdad(keys=["image"], func=lambda x: (x - 0.5) * 2),
        EnsureTyped(keys=["image"], dtype=dtype),
    ]

    if mode == "training":
        transforms = (
            Compose(base_transforms + post_transforms + final_transforms)
            if augment
            else Compose(base_transforms + post_transforms + final_transforms)
        )
    elif mode == "val":
        transforms = Compose(base_transforms + post_transforms + final_transforms)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return transforms


def prepare_ctvol_2019_10_dataset(ctvol, pixel_bounds, data_augment, num_channels, crop_type):
    """Run the older CT-volume preparation path used by legacy experiments."""
    max_slices = 402
    max_side_length = 420
    assert num_channels == 3 or num_channels == 1
    assert crop_type == "single"

    ctvol = pad_volume(ctvol, max_slices, max_side_length)

    if crop_type == "single":
        if data_augment is True:
            ctvol = single_crop_3d_augment(ctvol, max_slices, max_side_length)
        else:
            ctvol = single_crop_3d_fixed(ctvol, max_slices, max_side_length)
        if num_channels == 3:
            ctvol = reshape_3_channels(ctvol)
        output = torchify_pixelnorm_pixelcenter(ctvol, pixel_bounds)

    return output
