import torch
import torch.nn as nn
import torch.utils.data
import os
from pathlib import Path
import nibabel
import numpy as np
import logging
import tqdm
from datasets.utils import ColoredFormatter

"""LIDC dataset loader.

This module loads preprocessed LIDC-IDRI volumes referenced by a split text
file. Each split entry may be:
- an absolute path to `processed.nii.gz`
- a relative path
- a filename-like path resolvable from the configured dataset directory

The loader:
- resolves split entries robustly
- loads NIfTI volumes into memory
- converts them into channel-first tensors
- downsamples from 256 to 128 or 64 when requested
- applies optional normalization and lightweight augmentation

This matches the repository workflow where preprocessing happens first and
training consumes paths listed in `train.txt` and `val.txt`.
"""


class LIDCVolumes(torch.utils.data.Dataset):
    def __init__(
        self,
        directory,
        split_file,
        test_flag=False,
        normalize=None,
        mode="train",
        img_size=256,
        rank=0,
        augment=False,
    ):
        """
        Args:
            directory: Root directory for resolving relative split entries
            split_file: Path to the txt file containing image paths
            mode: 'train' or 'val'
            img_size: Target image size (64, 128, or 256)
        """
        super().__init__()

        assert img_size in [64, 128, 256], "Supported image sizes: 64, 128, 256"

        self.rank = rank
        self.mode = mode
        self.logger = self._setup_logger(rank)
        self.directory = str(Path(directory).expanduser())
        self.split_path = str(Path(split_file).expanduser())
        self.mode = mode
        self.img_size = img_size
        self.augment = augment
        self.normalize = normalize if normalize is not None else (lambda x: 2 * x - 1)
        self.data_cache = {}
        self.database = []

        if rank == 0:
            self.logger.info(f"Initializing LIDC dataset in {mode} mode")
            self.logger.info(f"Data directory: {self.directory}")
            self.logger.info(f"Loading paths from: {self.split_path}")

        if not os.path.exists(self.split_path):
            raise FileNotFoundError(f"Split file not found: {self.split_path}")

        with open(self.split_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        resolved_paths = self._resolve_split_entries(lines)
        self.database = [{"image": path} for path in resolved_paths]

        if rank == 0:
            self.logger.info(f"Loaded {len(self.database)} samples for {mode}")

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

    def _resolve_split_entries(self, entries):
        """Resolve absolute, relative, and filename-only entries from a split file."""
        resolved = []
        split_dir = os.path.dirname(self.split_path)
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

                if out.ndim == 3:
                    image = torch.zeros(1, 256, 256, 256)
                    image[0, :, :, :] = out
                else:
                    image = out

                if self.img_size == 128:
                    image = nn.AvgPool3d(2)(image.unsqueeze(0)).squeeze(0)
                elif self.img_size == 64:
                    image = nn.AvgPool3d(4)(image.unsqueeze(0)).squeeze(0)

                self.data_cache[name] = self.normalize(image)

            except Exception as e:
                if self.rank == 0:
                    self.logger.error(f"Error loading {name}: {e}")
                raise

    def __getitem__(self, index):
        """Return one cached volume sample, with optional augmentation."""
        name = self.database[index]["image"]
        image = self.data_cache[name]

        if self.augment and np.random.rand() > 0.5:
            image = torch.flip(image, dims=[-1])

        return {"image": image}

    def __len__(self):
        """Return the number of samples in the selected split."""
        return len(self.database)