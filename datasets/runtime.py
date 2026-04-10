"""Dataset runtime helpers shared across loaders.

Provides:

- ``get_rank_logger``: create a rank-aware logger without duplicate handlers
- ``build_train_val_split``: deterministic train/val split for an item list
- ``read_split_file`` / ``write_split_file``: plain-text split I/O
- ``write_split_metadata``: persist split configuration as JSON
- Default directory accessors for split files and preprocessed datasets
"""

import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from datasets.utils import ColoredFormatter


def get_default_split_dir() -> str:
    """Return the repository-level default directory for train/val split files."""
    return str(Path(__file__).resolve().parents[1] / "data")


def get_default_rad_chest_preprocessed_dir() -> str:
    """Return the repository-level default directory for preprocessed RAD-ChestCT volumes."""
    return str(Path(__file__).resolve().parents[1] / "data" / "rad_chestCT_preprocessed")


def get_default_lidc_preprocessed_dir() -> str:
    """Return the repository-level default directory for preprocessed LIDC volumes."""
    return str(Path(__file__).resolve().parents[1] / "data" / "lidc_preprocessed")


def get_rank_logger(name: str, rank: int) -> logging.Logger:
    """Create a rank-aware logger without duplicate handlers."""
    if rank != 0:
        logger = logging.getLogger(f"{name}.silent")
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger

    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def build_train_val_split(items: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """Build a deterministic train/val split from a list of items."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if not items:
        raise ValueError("Cannot build a split from an empty item list.")

    ordered_items = list(items)
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered_items)

    val_size = int(len(ordered_items) * val_ratio)
    val_items = ordered_items[:val_size]
    train_items = ordered_items[val_size:]
    return train_items, val_items


def read_split_file(path: str) -> List[str]:
    """Read a split text file and return non-empty stripped lines.

    Parameters
    ----------
    path : str
        Path to the split ``.txt`` file.

    Returns
    -------
    list of str
        Non-empty lines with leading/trailing whitespace removed.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


def write_split_file(path: str, values: Iterable[str]) -> None:
    """Write an iterable of strings to a split text file, one per line.

    Parameters
    ----------
    path : str
        Destination file path.
    values : iterable of str
        Items to write; each is followed by a newline.
    """
    with open(path, "w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def write_split_metadata(path: str, seed: int, val_ratio: float, train_count: int, val_count: int) -> None:
    """Persist split configuration metadata as a JSON file.

    Parameters
    ----------
    path : str
        Destination ``.json`` file path.
    seed : int
        Random seed used for the split.
    val_ratio : float
        Fraction of items assigned to the validation split.
    train_count : int
        Number of training items.
    val_count : int
        Number of validation items.
    """
    payload = {
        "seed": seed,
        "val_ratio": val_ratio,
        "train_count": train_count,
        "val_count": val_count,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
