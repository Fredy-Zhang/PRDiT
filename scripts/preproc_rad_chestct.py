"""RAD-ChestCT preprocessing script.

This script converts raw RAD-ChestCT `.npz` files into training-ready preprocessed
`.npz` volumes for this repository.

Expected input:
- A source directory containing raw `.npz` files.
- Each source file must contain a `ct` array representing a 3D CT volume.

Processing steps:
- Clip intensities to the fixed HU window `[-1000, 1000]`.
- Center crop the volume to `(256, 256, 256)`.
- Downsample from `(256, 256, 256)` to the requested target shape if needed.
- Optionally clip the upper intensity tail using a quantile threshold.
- Normalize the final cropped volume to `[0, 1]`.

Saved output:
- One compressed `.npz` file per source file in the target directory.
- Each saved file includes both `ct` and `volume` keys for compatibility.
- A text report summarizing processed, skipped, and failed files.
- Optional preview `.nii.gz` pairs for a small number of samples.

Typical usage:
- Preprocess a directory of raw RAD-ChestCT files before generating `train.txt`
  and `val.txt` with `scripts/split_train_val.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


LOW_HU = -1000.0
HIGH_HU = 1000.0
CANONICAL_CROP_SHAPE = (256, 256, 256)
DEFAULT_TARGET_SHAPE = (256, 256, 256)


def center_crop_3d(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray | None:
    """Center crop a 3D volume. Return None if any dimension is too small."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}.")

    if any(size < target for size, target in zip(volume.shape, target_shape)):
        return None

    starts = [(size - target) // 2 for size, target in zip(volume.shape, target_shape)]
    slices = tuple(slice(start, start + target) for start, target in zip(starts, target_shape))
    return volume[slices]


def resize_volume(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Resize a volume to target shape using trilinear interpolation."""
    factors = [float(target) / float(size) for target, size in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1, mode="constant", cval=LOW_HU)


def preprocess_volume(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    upper_quantile: float,
) -> np.ndarray:
    """Preprocess a RAD-ChestCT volume into a normalized ``[0, 1]`` array.

    Parameters
    ----------
    volume : numpy.ndarray
        Raw CT array (arbitrary dtype); HU values expected.
    target_shape : tuple of int
        ``(D, H, W)`` of the output volume.  A center crop to
        ``(256, 256, 256)`` is always applied first, followed by
        downsampling when ``target_shape`` differs.
    upper_quantile : float
        Fraction in ``(0, 1]`` used to clip upper-tail outliers before
        normalization.  Pass ``1.0`` to skip clipping.

    Returns
    -------
    numpy.ndarray, shape target_shape, dtype float32
        Volume with values in ``[0, 1]``.
    """
    volume = np.asarray(volume, dtype=np.float32)
    volume = np.clip(volume, LOW_HU, HIGH_HU)

    cropped = center_crop_3d(volume, CANONICAL_CROP_SHAPE)
    if cropped is None:
        raise ValueError(
            f"Volume shape {volume.shape} is smaller than canonical crop {CANONICAL_CROP_SHAPE}."
        )

    if target_shape == CANONICAL_CROP_SHAPE:
        processed = cropped
    else:
        processed = resize_volume(cropped, target_shape=target_shape).astype(np.float32)

    if not 0.0 < upper_quantile <= 1.0:
        raise ValueError(f"`upper_quantile` must be in (0, 1], got {upper_quantile}.")

    if upper_quantile < 1.0:
        upper_clip = np.quantile(processed, upper_quantile)
        processed = np.clip(processed, LOW_HU, upper_clip)

    min_value = float(processed.min())
    max_value = float(processed.max())
    if max_value <= min_value:
        raise ValueError(
            f"Volume becomes constant after preprocessing (min={min_value}, max={max_value})."
        )

    processed = (processed - min_value) / (max_value - min_value)
    return processed.astype(np.float32)


def process_file(
    source_path: Path,
    target_path: Path,
    target_shape: tuple[int, int, int],
    upper_quantile: float,
) -> tuple[bool, str, np.ndarray | None, np.ndarray | None]:
    """Preprocess one source file and save the converted volume to the target path."""
    if target_path.exists():
        return True, "skipped_existing", None, None

    data = np.load(source_path)
    if "ct" not in data:
        raise KeyError(f"Missing `ct` in {source_path}. Available keys: {list(data.keys())}")

    volume = data["ct"].astype(np.float32)
    processed = preprocess_volume(
        volume=volume,
        target_shape=target_shape,
        upper_quantile=upper_quantile,
    )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target_path,
        ct=processed,
        volume=processed,
        source_file=str(source_path.resolve()),
        original_shape=np.asarray(volume.shape, dtype=np.int32),
    )
    return True, "processed", volume, processed


def collect_npz_files(source_dir: Path, recursive: bool) -> list[Path]:
    """Collect raw `.npz` files from the source directory."""
    pattern = "**/*.npz" if recursive else "*.npz"
    return sorted(path for path in source_dir.glob(pattern) if path.is_file())


def write_lines(path: Path, lines: list[str]) -> None:
    """Write plain-text report lines to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if lines:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")


def save_nifti(volume: np.ndarray, path: Path) -> None:
    """Save a numpy volume as a `.nii.gz` file for quick visual inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(volume.astype(np.float32), affine=np.eye(4))
    nib.save(nii, str(path))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for RAD-ChestCT preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess RAD-ChestCT raw .npz files into normalized training volumes.",
        epilog=(
            "Example:\n"
            "  python scripts/preprocess_rad_chestct.py "
            "--source_dir rad_chestCT "
            "--target_dir rad_chestct "
            "--target-shape 128 128 128"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory containing raw RAD-ChestCT .npz files with a `ct` array.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Directory where preprocessed .npz files will be written.",
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=DEFAULT_TARGET_SHAPE,
        metavar=("D", "H", "W"),
        help="Output volume shape.",
    )
    parser.add_argument(
        "--upper-quantile",
        type=float,
        default=0.995,
        help="Optional upper-tail intensity clipping before normalization. Use 1.0 to disable.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan `source_dir` for .npz files.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path for a processing report. Defaults to <target_dir>/preprocess_report.txt.",
    )
    parser.add_argument(
        "--save-preview-count",
        type=int,
        default=5,
        help="Number of successfully processed samples to also export as .nii.gz for QC.",
    )
    return parser


def main() -> None:
    """Run the RAD-ChestCT preprocessing pipeline from command-line arguments."""
    args = build_parser().parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()
    target_shape = tuple(args.target_shape)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    input_files = collect_npz_files(source_dir, recursive=args.recursive)
    if not input_files:
        raise FileNotFoundError(f"No .npz files found in {source_dir}")

    log_path = (
        Path(args.log_file).expanduser().resolve()
        if args.log_file is not None
        else target_dir / "preprocess_report.txt"
    )

    processed_count = 0
    skipped_existing_count = 0
    preview_saved_count = 0
    failures: list[str] = []
    report_lines = [
        f"Source directory: {source_dir}",
        f"Target directory: {target_dir}",
        f"Input files found: {len(input_files)}",
        "Image key: ct",
        f"Canonical crop shape: {CANONICAL_CROP_SHAPE}",
        f"Target shape: {target_shape}",
        "Resize policy: crop to canonical shape, then downsample if needed",
        f"Upper quantile: {args.upper_quantile}",
        "",
    ]

    for source_path in tqdm(input_files, desc="Preprocessing RAD-ChestCT", ncols=100):
        relative_path = source_path.relative_to(source_dir)
        target_path = target_dir / relative_path

        try:
            ok, status, original_volume, processed_volume = process_file(
                source_path=source_path,
                target_path=target_path,
                target_shape=target_shape,
                upper_quantile=args.upper_quantile,
            )
            if ok and status == "processed":
                processed_count += 1
                if preview_saved_count < args.save_preview_count:
                    stem = source_path.stem
                    preview_root = target_dir / "preview_nii"
                    save_nifti(
                        original_volume,
                        preview_root / "original" / f"{stem}.nii.gz",
                    )
                    save_nifti(
                        processed_volume,
                        preview_root / "processed" / f"{stem}.nii.gz",
                    )
                    preview_saved_count += 1
            elif ok and status == "skipped_existing":
                skipped_existing_count += 1
        except Exception as exc:
            failures.append(f"{source_path}: {exc}")

    report_lines.extend(
        [
            f"Processed files: {processed_count}",
            f"Skipped existing: {skipped_existing_count}",
            f"Preview NIfTI pairs saved: {preview_saved_count}",
            f"Failed files: {len(failures)}",
            "",
        ]
    )
    if failures:
        report_lines.append("Failures:")
        report_lines.extend(failures)

    write_lines(log_path, report_lines)

    print("RAD-ChestCT preprocessing finished.")
    print(f" - input files: {len(input_files)}")
    print(f" - processed: {processed_count}")
    print(f" - skipped existing: {skipped_existing_count}")
    print(f" - preview nii.gz pairs saved: {preview_saved_count}")
    print(f" - failed: {len(failures)}")
    print(f" - report: {log_path}")


if __name__ == "__main__":
    main()
