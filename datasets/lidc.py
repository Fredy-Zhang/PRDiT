"""LIDC dataset loader.

This module loads preprocessed LIDC-IDRI volumes referenced by a split text
file. Each split entry may be:

- an absolute path to ``processed.nii.gz``
- a relative path
- a filename-like path resolvable from the configured dataset directory

The loader:

- resolves split entries robustly
- loads NIfTI volumes into memory
- converts them into channel-first tensors
- downsamples from 256 to 128 or 64 when requested
- applies optional normalization and lightweight augmentation

This matches the repository workflow where preprocessing happens first and
training consumes paths listed in ``train.txt`` and ``val.txt``.
"""

import logging
import os
from pathlib import Path

import nibabel
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import tqdm

from datasets.utils import ColoredFormatter


class LIDCVolumes(torch.utils.data.Dataset):
    def __init__(
        self,
        directory,
        split_file=None,
        normalize=None,
        mode="train",
        img_size=256,
        split_dir=None,
        rank=0,
        augment=False,
    ):
        """Initialize the LIDC-IDRI dataset.

        Parameters
        ----------
        directory : str or Path
            Root directory for resolving relative split entries, or the
            directory of generated volumes when ``mode='fake'``.
        split_file : str or list of str, optional
            Path to the ``.txt`` split file for ``mode='train'`` / ``'val'``,
            or a list of paths to merge for ``mode='real'``.
            Takes precedence over ``split_dir``.
        normalize : callable or None, optional
            Voxel normalization applied after loading; ``None`` defaults to
            ``lambda x: 2*x - 1``.
        mode : str, optional
            One of ``'train'``, ``'val'``, ``'real'``, or ``'fake'``
            (default ``'train'``).
        img_size : int, optional
            Target voxel edge length — 64, 128, or 256 (default ``256``).
        split_dir : str or Path, optional
            Directory containing ``train.txt`` and ``val.txt`` used when
            ``mode='real'`` and ``split_file`` is not given.
        rank : int, optional
            Process rank; only rank 0 emits progress output (default ``0``).
        augment : bool, optional
            Apply random left-right flip augmentation (default ``False``).
        """
        super().__init__()

        assert img_size in [64, 128, 256], "Supported image sizes: 64, 128, 256"
        assert mode in ("train", "val", "real", "fake"), \
            "mode must be 'train', 'val', 'real', or 'fake'"

        self.rank = rank
        self.mode = mode
        self.logger = self._setup_logger(rank)
        self.directory = str(Path(directory).expanduser())
        self.img_size = img_size
        self.augment = augment
        # Real LIDC volumes are stored in [0, 1] and mapped to [-1, 1] via 2x-1.
        # Generated ('fake') volumes are already saved in the model's [-1, 1]
        # space, so re-applying 2x-1 would double-normalize them; use identity.
        if normalize is not None:
            self.normalize = normalize
        elif mode == "fake":
            # Already in [-1, 1]; clamp the generator's slight overshoot back to
            # the physical range that real (normalized) CT occupies.
            self.normalize = lambda x: x.clamp(-1.0, 1.0)
        else:
            self.normalize = lambda x: 2 * x - 1
        self.data_cache = {}
        self.database = []

        if rank == 0:
            self.logger.info(f"Initializing LIDC dataset in '{mode}' mode")
            self.logger.info(f"Directory: {self.directory}")

        if mode == "fake":
            # Scan all .nii.gz files directly in the directory
            paths = sorted(str(p) for p in Path(self.directory).glob("*.nii.gz"))
            if not paths:
                raise FileNotFoundError(
                    f"No .nii.gz files found in fake directory: {self.directory}"
                )
            self.database = [{"image": p, "name": os.path.basename(p)} for p in paths]
        elif mode == "real":
            # Resolve split files: explicit list > split_dir/train.txt + split_dir/val.txt
            if not split_file:
                if split_dir is None:
                    raise ValueError(
                        "For mode='real', provide either split_dir (directory containing "
                        "train.txt and val.txt) or an explicit split_file list."
                    )
                split_dir = str(Path(split_dir).expanduser())
                split_file = [
                    os.path.join(split_dir, "train.txt"),
                    os.path.join(split_dir, "val.txt"),
                ]
            paths = self._load_real_splits(split_file)
            self.database = [{"image": p} for p in paths]
        else:
            # 'train' or 'val': use the provided split file
            if split_file is None:
                raise ValueError("split_file must be provided for mode='train' or 'val'.")
            self.split_path = str(Path(split_file).expanduser())
            if not os.path.exists(self.split_path):
                raise FileNotFoundError(f"Split file not found: {self.split_path}")
            with open(self.split_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            paths = self._resolve_split_entries(lines)
            self.database = [{"image": p} for p in paths]

        if rank == 0:
            self.logger.info(f"Loaded {len(self.database)} samples")

        self._preload_data()

    def _setup_logger(self, rank):
        """Create a rank-aware logger for dataset initialization and loading."""
        logger_name = f"LIDCVolumes_{self.mode}"
        logger = logging.getLogger(logger_name)
        logger.propagate = False
        if not logger.handlers and rank == 0:
            handler = logging.StreamHandler()
            formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        elif rank != 0:
            logger.addHandler(logging.NullHandler())
        return logger

    def _load_real_splits(self, split_files):
        """Merge multiple split files to build the full ground-truth file list."""
        entries = []
        for path in split_files:
            path = str(Path(path).expanduser())
            if not os.path.exists(path):
                raise FileNotFoundError(f"Split file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                loaded = [line.strip() for line in f if line.strip()]
            entries.extend(loaded)
            self.logger.info(f"Loaded {len(loaded)} entries from {path}")
        if not entries:
            raise ValueError(f"All provided split files are empty: {split_files}")
        return self._resolve_split_entries(entries)

    def _resolve_split_entries(self, entries):
        """Resolve absolute, relative, and filename-only entries from a split file."""
        resolved = []
        split_dir = os.path.dirname(getattr(self, "split_path", self.directory))
        for entry in entries:
            if os.path.isabs(entry):
                resolved.append(entry)
                continue

            directory_candidate = os.path.join(self.directory, entry)
            split_dir_candidate = os.path.join(split_dir, entry)

            if os.path.exists(directory_candidate):
                resolved.append(directory_candidate)
            elif os.path.exists(split_dir_candidate):
                resolved.append(split_dir_candidate)
            else:
                resolved.append(directory_candidate)

        missing = [path for path in resolved if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                f"Split file {self.split_path} references missing files. "
                f"First missing file: {missing[0]}"
            )
        return resolved

    def _preload_data(self):
        """Load all referenced NIfTI volumes into memory and apply optional normalization."""
        if self.rank == 0:
            self.logger.info(f"Preloading {self.mode} data into memory...")

        for filedict in tqdm.tqdm(self.database, desc=f"Loading {self.mode}", disable=(self.rank != 0)):
            name = filedict["image"]
            try:
                nib_img = nibabel.load(name)
                out = torch.from_numpy(nib_img.get_fdata()).float()

                image = out.unsqueeze(0) if out.ndim == 3 else out

                # Downsample from the volume's native resolution to img_size.
                # Real LIDC volumes are stored at 256^3; generated samples are
                # already at the model's img_size, so the factor may be 1.
                src = image.shape[-1]
                if src % self.img_size != 0:
                    raise ValueError(
                        f"Volume edge {src} is not a multiple of img_size {self.img_size}: {name}"
                    )
                factor = src // self.img_size
                if factor > 1:
                    image = nn.AvgPool3d(factor)(image.unsqueeze(0)).squeeze(0)

                self.data_cache[name] = self.normalize(image)

            except Exception as e:
                if self.rank == 0:
                    self.logger.error(f"Error loading {name}: {e}")
                raise

    def __getitem__(self, index):
        """Return one cached volume sample, with optional augmentation."""
        entry = self.database[index]
        name = entry["image"]
        image = self.data_cache[name]

        if self.augment and np.random.rand() > 0.5:
            image = torch.flip(image, dims=[-1])

        if self.mode == "fake":
            return image, entry["name"]
        return {"image": image}

    def __len__(self):
        """Return the number of samples in the selected split."""
        return len(self.database)