"""Local denoiser components for PRDiT.

This module groups the coarse local denoising path used in stage-1 training:

- `ExtractPatches3D`: convert input volumes into patch tokens
- `MlpDenoiser`: denoise patch tokens directly in patch space
- `CoarseDenoiser`: end-to-end coarse branch that applies extraction + denoising
"""

from __future__ import annotations

from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import SwiGLU

from models.utils import _ntuple, modulate


to_3tuple = _ntuple(3)


class ExtractPatches3D(nn.Module):
    """Extract 3D patches from a volume and flatten them into a token sequence.

    Uses chained ``unfold`` operations to support both overlapping and
    non-overlapping patches.  Output shape is ``[B, N, patch_dim]``.

    Parameters
    ----------
    patch_size : int or tuple of int
        Edge length of extracted patches (scalar or 3-tuple).
    stride : int or tuple of int
        Stride for patch extraction (scalar or 3-tuple).
    padding : int, optional
        Reflection padding applied before extraction (default ``0``).
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]],
        padding: int = 0,
    ):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.stride = to_3tuple(stride)
        self.padding = padding

    def compute_num_patches(
        self,
        input_size: Union[int, Tuple[int, int, int]],
    ) -> Tuple[int, Tuple[int, int, int]]:
        """Return the number of extracted patches and the 3D patch grid shape."""
        input_size = to_3tuple(input_size)
        if self.padding > 0:
            input_size = tuple(s + 2 * self.padding for s in input_size)

        grid_size = tuple(
            ((s - p) // st) + 1
            for s, p, st in zip(input_size, self.patch_size, self.stride)
        )
        return grid_size[0] * grid_size[1] * grid_size[2], grid_size

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Extract and flatten 3D patches from an input volume."""
        batch_size, channels, _, _, _ = volume.size()

        if self.padding > 0:
            volume = F.pad(volume, (self.padding,) * 6, mode="reflect")

        patches = (
            volume.unfold(2, self.patch_size[0], self.stride[0])
            .unfold(3, self.patch_size[1], self.stride[1])
            .unfold(4, self.patch_size[2], self.stride[2])
        )

        patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        num_patches = patches.numel() // (batch_size * channels * patch_volume)

        patches = (
            patches.contiguous()
            .view(batch_size, channels, num_patches, patch_volume)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, num_patches, -1)
        )
        return patches

    def extra_repr(self) -> str:
        """String representation of module parameters."""
        return f"patch_size={self.patch_size}, stride={self.stride}, padding={self.padding}"


class MlpDenoiser(nn.Module):
    """Patchwise MLP denoiser used in the coarse PRDiT path.

    Performs lightweight denoising directly in patch space using two SwiGLU
    blocks with timestep-conditioned modulation.

    Notes
    -----
    ``hidden_size`` refers to the patch-token width
    (``in_channels * extract_patch_size**3``), not the transformer hidden size.
    ``input_size``, ``act_layer``, and ``swiglu_mlp`` are retained only for
    API compatibility with older constructor signatures.

    Parameters
    ----------
    input_size : int
        Target cubic volume size (retained for compatibility; unused).
    hidden_size : int
        Patch-token feature dimension.
    patch_size : int
        Edge length of each output patch.
    out_channels : int
        Number of output channels predicted per patch.
    act_layer : callable, optional
        Retained for compatibility (default ``nn.ReLU``).
    mlp_ratio : float, optional
        Hidden layer expansion ratio inside the SwiGLU blocks (default ``1.0``).
    swiglu_mlp : bool, optional
        Retained for compatibility (default ``False``).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        act_layer: Callable = nn.ReLU,
        mlp_ratio: float = 1.0,
        swiglu_mlp: bool = False,
    ):
        super().__init__()

        del input_size, act_layer, swiglu_mlp
        token_dim = hidden_size

        self.norm1 = nn.LayerNorm(token_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(token_dim, elementwise_affine=False, eps=1e-6)

        self.mlp1 = SwiGLU(
            in_features=token_dim,
            hidden_features=int(token_dim * mlp_ratio),
            norm_layer=nn.LayerNorm,
            drop=0,
        )
        self.mlp2 = SwiGLU(
            in_features=token_dim,
            hidden_features=int(token_dim * mlp_ratio),
            norm_layer=nn.LayerNorm,
            drop=0,
        )

        self.linear_final = nn.Linear(token_dim, patch_size**3 * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(token_dim, 6 * token_dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Denoise patch tokens with timestep-conditioned MLP blocks."""
        shift1, scale1, shift2, scale2, shift3, scale3 = self.adaLN_modulation(c).chunk(6, dim=1)

        h = self.mlp1(modulate(self.norm1(x), shift1, scale1))
        h = self.mlp2(modulate(self.norm2(h), shift2, scale2))
        h = h + x
        h = modulate(h, shift3, scale3)
        return self.linear_final(h)


class CoarseDenoiser(nn.Module):
    """Coarse denoising path using an MLP operating directly in patch space.

    Extracts raw 3D patches from the input volume and denoises them with
    :class:`MlpDenoiser`.

    Notes
    -----
    ``hidden_size`` is retained for constructor compatibility with the full
    PRDiT config; the actual coarse MLP width is ``in_channels * extract_patch_size**3``.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input volume.
    extract_patch_size : int
        Edge length of the patches extracted from the input.
    hidden_size : int
        Retained for PRDiT config compatibility (unused).
    patch_size : int
        Edge length of each output patch.
    out_channels : int
        Number of channels predicted per voxel.
    input_size : int
        Edge length of the cubic input/output volume.
    stride : int, optional
        Patch extraction stride (default ``4``).
    padding : int, optional
        Reflection padding used before extraction (default ``2``).
    mlp_ratio : float, optional
        SwiGLU expansion ratio (default ``1.0``).
    swiglu_mlp : bool, optional
        Retained for compatibility (default ``True``).
    act_layer : callable, optional
        Retained for compatibility (default ``nn.GELU``).
    """

    def __init__(
        self,
        in_channels: int,
        extract_patch_size: int,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        input_size: int,
        stride: int = 4,
        padding: int = 2,
        mlp_ratio: float = 1.0,
        swiglu_mlp: bool = True,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        del hidden_size

        self.patch_extractor = ExtractPatches3D(
            patch_size=extract_patch_size,
            stride=stride,
            padding=padding,
        )

        self.num_patches, self.grid_size = self.patch_extractor.compute_num_patches(input_size)

        patch_token_dim = in_channels * extract_patch_size**3
        output_volume_size = input_size
        output_patch_size = patch_size

        self.mlp_denoise = MlpDenoiser(
            input_size=output_volume_size,
            hidden_size=patch_token_dim,
            patch_size=output_patch_size,
            out_channels=out_channels,
            swiglu_mlp=swiglu_mlp,
            act_layer=act_layer,
            mlp_ratio=mlp_ratio,
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        return_patches: bool = False,
    ) -> torch.Tensor:
        """Run the coarse branch on an input volume."""
        patches = self.patch_extractor(x)
        patch_predictions = self.mlp_denoise(patches, c)

        if return_patches:
            return patches, patch_predictions
        return patch_predictions
