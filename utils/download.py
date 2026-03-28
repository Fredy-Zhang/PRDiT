# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for downloading or loading pre-trained DiT checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.datasets.utils import download_url


PRETRAINED_MODELS = {"DiT-XL-2-512x512.pt", "DiT-XL-2-256x256.pt"}
PRETRAINED_MODEL_DIR = Path("pretrained_models")
PRETRAINED_MODEL_BASE_URL = "https://dl.fbaipublicfiles.com/DiT/models"


def _load_checkpoint(path: Path):
    """Load a checkpoint file and return the EMA weights when present."""
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        return checkpoint["ema"]
    return checkpoint


def find_model(model_name: str):
    """Load a known pre-trained model or a user-provided local checkpoint."""
    if model_name in PRETRAINED_MODELS:
        return download_model(model_name)

    checkpoint_path = Path(model_name).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Could not find DiT checkpoint at {checkpoint_path}")
    return _load_checkpoint(checkpoint_path)


def download_model(model_name: str):
    """Download a pre-trained DiT checkpoint if needed and return its weights."""
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown pre-trained model: {model_name}")

    PRETRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local_path = PRETRAINED_MODEL_DIR / model_name
    if not local_path.is_file():
        web_path = f"{PRETRAINED_MODEL_BASE_URL}/{model_name}"
        download_url(web_path, str(PRETRAINED_MODEL_DIR))
    return _load_checkpoint(local_path)


if __name__ == "__main__":
    for model_name in sorted(PRETRAINED_MODELS):
        download_model(model_name)
    print("Done.")
