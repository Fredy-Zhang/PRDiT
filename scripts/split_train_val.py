"""Train/validation split generation utilities.

This script creates `train.txt` and `val.txt` files for the datasets used in
this repository.

Supported dataset layouts:
- LIDC-IDRI:
  A root directory containing subfolders like `LIDC-IDRI-XXXX`, where each
  subfolder contains `processed.nii.gz`.
- RAD-ChestCT:
  A flat directory containing preprocessed `.npz` files such as `trn07793.npz`.

Output format:
- One sample path per line.
- Paths are written as absolute paths so training can run from any working
  directory.

Split policy:
- Deterministic shuffling controlled by a random seed.
- Either a validation ratio or a fixed validation set size can be used.

Typical usage:
- Run this after preprocessing to generate `lidc_data/train.txt`,
  `lidc_data/val.txt`, `rad_data/train.txt`, or `rad_data/val.txt`.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

def collect_rad_paths(data_root: Path, suffix: str) -> list[Path]:
    """Collect files directly under `data_root` matching the RAD-ChestCT layout."""
    return sorted(
        path.resolve()
        for path in data_root.iterdir()
        if path.is_file() and path.suffix == suffix
    )


def collect_lidc_paths(data_root: Path, filename: str) -> list[Path]:
    """Collect `filename` from immediate `LIDC-IDRI-*` subdirectories."""
    paths: list[Path] = []

    for entry in sorted(data_root.iterdir(), key=lambda path: path.name):
        if not entry.is_dir():
            continue
        if not entry.name.startswith("LIDC-IDRI-"):
            continue

        file_path = entry / filename
        if file_path.is_file():
            paths.append(file_path.resolve())

    return paths


def split_paths(
    paths: list[Path],
    seed: int,
    val_ratio: float | None = None,
    val_size: int | None = None,
) -> tuple[list[Path], list[Path]]:
    """Shuffle paths deterministically and split them into train and validation subsets."""
    if len(paths) < 2:
        raise ValueError(
            f"Need at least 2 samples to create train/val splits, found {len(paths)}."
        )

    if (val_ratio is None) == (val_size is None):
        raise ValueError("Provide exactly one of `val_ratio` or `val_size`.")

    shuffled_paths = list(paths)
    random.Random(seed).shuffle(shuffled_paths)

    if val_size is not None:
        if val_size <= 0:
            raise ValueError(f"`val_size` must be positive, got {val_size}.")
        if val_size >= len(shuffled_paths):
            raise ValueError(
                f"`val_size` must be smaller than the dataset size ({len(shuffled_paths)}), "
                f"got {val_size}."
            )
        val_count = val_size
    else:
        assert val_ratio is not None
        if not 0.0 < val_ratio < 1.0:
            raise ValueError(f"`val_ratio` must be between 0 and 1, got {val_ratio}.")
        val_count = max(1, int(round(len(shuffled_paths) * val_ratio)))

    train_count = len(shuffled_paths) - val_count
    if train_count == 0:
        raise ValueError(
            "Validation ratio leaves no training samples. Use a smaller `--val-ratio`."
        )

    train_set = shuffled_paths[:train_count]
    val_set = shuffled_paths[train_count:]
    return train_set, val_set


def write_split_file(output_path: Path, paths: list[Path]) -> None:
    """Write one sample path per line into a split text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(str(path) for path in paths) + "\n",
        encoding="utf-8",
    )


def generate_lidc_splits(
    data_root: str | Path,
    output_dir: str | Path,
    val_ratio: float | None = 0.1,
    val_size: int | None = None,
    seed: int = 42,
    filename: str = "processed.nii.gz",
    dataset: str = "auto",
) -> tuple[Path, Path]:
    """Generate ``train.txt`` and ``val.txt`` for LIDC-IDRI or RAD-ChestCT.

    Auto-detects the dataset layout when ``dataset='auto'``: checks for
    ``LIDC-IDRI-*`` subdirectories first, then falls back to flat ``.npz``
    files.

    Parameters
    ----------
    data_root : str or Path
        Root directory of the preprocessed dataset.
    output_dir : str or Path
        Directory where ``train.txt`` and ``val.txt`` are written.
    val_ratio : float or None, optional
        Fraction of samples assigned to validation (default ``0.1``).
        Mutually exclusive with ``val_size``.
    val_size : int or None, optional
        Exact number of validation samples. Mutually exclusive with
        ``val_ratio``.
    seed : int, optional
        Random seed for deterministic shuffling (default ``42``).
    filename : str, optional
        Volume filename expected inside each ``LIDC-IDRI-*`` directory
        (default ``"processed.nii.gz"``).
    dataset : str, optional
        Layout hint — ``"auto"``, ``"lidc"``, or ``"rad"``
        (default ``"auto"``).

    Returns
    -------
    train_txt : Path
        Path to the written training split file.
    val_txt : Path
        Path to the written validation split file.
    """
    data_root = Path(data_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {data_root}")

    if dataset == "lidc":
        all_paths = collect_lidc_paths(data_root, filename)
        dataset_name = "lidc"
    elif dataset == "rad":
        all_paths = collect_rad_paths(data_root, suffix=".npz")
        dataset_name = "rad"
    elif dataset == "auto":
        lidc_paths = collect_lidc_paths(data_root, filename)
        if lidc_paths:
            all_paths = lidc_paths
            dataset_name = "lidc"
        else:
            rad_paths = collect_rad_paths(data_root, suffix=".npz")
            all_paths = rad_paths
            dataset_name = "rad" if rad_paths else "unknown"
    else:
        raise ValueError(f"Unsupported dataset type: {dataset}")

    print(f"Found {len(all_paths)} samples under {data_root}.")

    if not all_paths:
        raise FileNotFoundError(
            f"Could not detect supported dataset files in {data_root}. "
            f"Expected either `LIDC-IDRI-*/{filename}` or flat `.npz` files."
        )

    train_set, val_set = split_paths(
        all_paths,
        seed=seed,
        val_ratio=val_ratio,
        val_size=val_size,
    )

    train_txt = output_dir / "train.txt"
    val_txt = output_dir / "val.txt"
    write_split_file(train_txt, train_set)
    write_split_file(val_txt, val_set)

    print("Generated split files:")
    print(f" - train ({len(train_set)} samples): {train_txt}")
    print(f" - val   ({len(val_set)} samples): {val_txt}")
    print(f" - dataset: {dataset_name}")
    print(f" - seed: {seed}")
    if val_size is not None:
        print(f" - split mode: fixed validation size ({val_size})")
    else:
        print(f" - split mode: validation ratio ({val_ratio})")

    return train_txt, val_txt


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for generating split files."""
    parser = argparse.ArgumentParser(
        description="Generate train.txt and val.txt for LIDC-IDRI or RAD-ChestCT preprocessed volumes."
    )
    parser.add_argument(
        "data_root",
        nargs="?",
        default="data/LIDC-IDRI",
        help="Dataset root. Supports LIDC-IDRI-* subfolders or flat RAD-ChestCT .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        default="lidc_data",
        help="Directory where train.txt and val.txt will be written.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples to place in the validation split.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="Exact number of samples to place in the validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible shuffling.",
    )
    parser.add_argument(
        "--filename",
        default="processed.nii.gz",
        help="Volume filename expected inside each LIDC-IDRI-* directory.",
    )
    parser.add_argument(
        "--dataset",
        choices=("auto", "lidc", "rad"),
        default="auto",
        help="Dataset layout to use. `auto` detects LIDC vs flat RAD-ChestCT .npz files.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    val_ratio = None if args.val_size is not None else args.val_ratio
    generate_lidc_splits(
        data_root=args.data_root,
        output_dir=args.output_dir,
        val_ratio=val_ratio,
        val_size=args.val_size,
        seed=args.seed,
        filename=args.filename,
        dataset=args.dataset,
    )
