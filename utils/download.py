# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for loading PRDiT checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch


def _load_checkpoint(path: Path):
    """Load a checkpoint file and return the EMA weights when present."""
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        return checkpoint["ema"]
    return checkpoint


def find_model(model_name: str):
    """Load a user-provided local checkpoint."""
    checkpoint_path = Path(model_name).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Could not find PRDiT checkpoint at {checkpoint_path}")
    return _load_checkpoint(checkpoint_path)
