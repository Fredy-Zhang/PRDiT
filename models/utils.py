"""Utility functions shared across PRDiT model components.

This module contains small helpers for:
- tuple normalization
- adaptive feature modulation
- unpatchifying patch-space predictions back to dense 3D volumes
- 1D / 3D sinusoidal positional encodings
- stochastic depth
"""

from __future__ import annotations

import collections.abc
import logging
import math
from itertools import repeat
from typing import Any, Callable, Optional

import numpy as np
import torch


logger = logging.getLogger(__name__)

__all__ = [
    "_ntuple",
    "to_3tuple",
    "modulate",
    "unpatchify_3d",
    "get_normalized_3d_pos_enc",
    "get_1d_sincos_pos_embed_from_grid",
    "get_3d_sincos_pos_embed_from_grid",
    "get_3d_sincos_pos_embed",
    "drop_path",
]


def _ntuple(n: int) -> Callable[[Any], tuple[Any, ...]]:
    """Return a parser that converts scalars or iterables into an `n`-tuple."""

    def parse(x: Any) -> tuple[Any, ...]:
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_3tuple = _ntuple(3)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive shift-scale modulation to a token tensor."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def unpatchify_3d(
    x: torch.Tensor,
    out_channels: int,
    patch_size: int,
    input_size: int,
) -> torch.Tensor:
    """Reconstruct a dense 3D volume from patch-space predictions.

    Parameters
    ----------
    x : torch.Tensor
        Patch predictions of shape ``[B, N, patch_size³ * C]``.
    out_channels : int
        Number of output channels *C*.
    patch_size : int
        Edge length of each cubic patch.
    input_size : int
        Edge length of the reconstructed cubic volume.

    Returns
    -------
    torch.Tensor
        Reconstructed volume of shape ``[B, C, D, H, W]``.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape [B, N, patch_dim], got {tuple(x.shape)}")
    if input_size % patch_size != 0:
        raise ValueError(f"input_size ({input_size}) must be divisible by patch_size ({patch_size})")

    channels = out_channels
    patch_edge = patch_size
    grid_size = input_size // patch_edge
    expected_patch_dim = patch_edge**3 * channels

    if x.shape[-1] != expected_patch_dim:
        raise ValueError(
            f"Last dimension mismatch: got {x.shape[-1]}, expected {expected_patch_dim} "
            f"for patch_size={patch_edge} and out_channels={channels}"
        )

    x = x.reshape(-1, grid_size, grid_size, grid_size, patch_edge, patch_edge, patch_edge, channels)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return x.reshape(-1, channels, grid_size * patch_edge, grid_size * patch_edge, grid_size * patch_edge)


def get_normalized_3d_pos_enc(
    grid_size: int,
    embed_dim: int,
    num_frequencies: Optional[int] = None,
) -> torch.Tensor:
    """Generate normalized 3D sinusoidal positional encodings.

    Coordinates are normalized to ``[0, 1]`` and encoded with log-spaced
    frequency bands chosen to remain stable across spatial resolutions.

    Parameters
    ----------
    grid_size : int
        Edge length of the cubic spatial grid.
    embed_dim : int
        Output embedding dimension (must be divisible by 6 unless
        ``num_frequencies`` is given).
    num_frequencies : int or None, optional
        Number of frequency bands per spatial axis.  Inferred from
        ``embed_dim`` when ``None`` (default ``None``).

    Returns
    -------
    torch.Tensor
        Position encoding of shape ``(grid_size³, embed_dim)``.
    """
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive, got {embed_dim}")

    if num_frequencies is None:
        if embed_dim % 6 != 0:
            raise ValueError("embed_dim must be divisible by 6 or num_frequencies must be provided")
        num_frequencies = embed_dim // 6
    elif num_frequencies <= 0:
        raise ValueError(f"num_frequencies must be positive, got {num_frequencies}")

    logger.debug(
        "Generating normalized 3D position encoding: grid_size=%s embed_dim=%s num_frequencies=%s",
        grid_size,
        embed_dim,
        num_frequencies,
    )

    coords = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    pos = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    safety_factor = 0.95
    f_min, f_max = 1.0, safety_factor * (grid_size / 2.0)
    t = torch.linspace(0.0, 1.0, num_frequencies, dtype=torch.float32)
    freqs = f_min * (f_max / f_min) ** t * 2.0 * math.pi

    encodings = []
    for dim in range(3):
        for fn in (torch.sin, torch.cos):
            encodings.append(fn(pos[:, dim : dim + 1] * freqs))

    pos_enc = torch.cat(encodings, dim=-1)
    if pos_enc.shape[-1] != embed_dim:
        raise ValueError(f"Encoding dimension mismatch: got {pos_enc.shape[-1]}, expected {embed_dim}")
    return pos_enc


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sinusoidal positional embeddings from a 1D position grid.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension (must be even).
    pos : np.ndarray
        1-D array of position values.

    Returns
    -------
    np.ndarray
        Embedding matrix of shape ``(len(pos), embed_dim)``.
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_3d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generate 3D sinusoidal positional embeddings from a stacked 3D grid.

    Parameters
    ----------
    embed_dim : int
        Total embedding dimension (must be divisible by 3).
    grid : np.ndarray
        Stacked coordinate grid of shape ``[3, ...]``.

    Returns
    -------
    np.ndarray
        Concatenated per-axis embeddings of shape ``(N, embed_dim)``.
    """
    if embed_dim % 3 != 0:
        raise ValueError("embed_dim must be divisible by 3")

    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
    return np.concatenate([emb_x, emb_y, emb_z], axis=1)


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    extra_tokens: int = 0,
) -> np.ndarray:
    """Generate 3D sinusoidal positional embeddings on a cubic grid.

    Parameters
    ----------
    embed_dim : int
        Total embedding dimension (must be divisible by 3).
    grid_size : int
        Edge length of the cubic spatial grid.
    cls_token : bool, optional
        Prepend zero-filled class-token rows when ``True`` (default ``False``).
    extra_tokens : int, optional
        Number of extra zero-filled rows to prepend (default ``0``).

    Returns
    -------
    np.ndarray
        Positional embedding of shape ``(grid_size³ [+ extra_tokens], embed_dim)``.
    """
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")

    grid_x = np.arange(grid_size, dtype=np.float32)
    grid_y = np.arange(grid_size, dtype=np.float32)
    grid_z = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    """Apply per-sample stochastic depth to the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape ``[B, ...]``.
    drop_prob : float, optional
        Probability of dropping each sample's residual (default ``0.0``).
    training : bool, optional
        Whether the model is in training mode (default ``False``).
    scale_by_keep : bool, optional
        Scale kept samples by ``1 / keep_prob`` when ``True`` (default ``True``).

    Returns
    -------
    torch.Tensor
        Input tensor with stochastic depth applied.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor
