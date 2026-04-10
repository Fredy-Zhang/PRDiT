"""Specialized PRDiT output heads and coarse denoisers.

This module contains the experimental RMSNorm-based variants of the main model
heads used in PRDiT:

- `MlpDenoiserRMS`: a coarse patch denoiser with separate image and noise heads
- `FinalLayerRMS`: a final projection layer with the same dual-head structure

These classes mirror the structure of the corresponding implementations in
`models/models.py`, but keep the image and noise branches explicit so the image
head can use `RMSNorm` while the noise head uses `LayerNorm`.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
from timm.layers import SwiGLU

from models.classes import RMSNorm
from models.local_denoiser import ExtractPatches3D
from models.utils import modulate, unpatchify_3d


class MlpDenoiserRMS(nn.Module):
    """Dual-head patch MLP denoiser with RMSNorm on the image branch.

    This module is intended for the coarse PRDiT path when we want separate
    image and noise predictions instead of a single shared output projection.
    Two timestep-conditioned SwiGLU blocks refine the patch tokens, after which
    the features are split into:

    - an image head normalized with `RMSNorm`
    - a noise head normalized with `LayerNorm`

    When `out_channels == 2`, the module reconstructs dense 3D volumes and
    returns them concatenated as `[noise, image]` along the channel dimension.

    Notes
    -----
    ``hidden_size`` here refers to the coarse patch-token width
    (``in_channels * extract_patch_size**3``), not the transformer hidden size.
    ``swiglu_mlp`` is retained only for API compatibility.

    Parameters
    ----------
    input_size : int
        Cubic edge length of the target output volume.
    hidden_size : int
        Patch-token feature dimension.
    patch_size : int
        Edge length of each cubic output patch.
    out_channels : int
        Number of output channels; ``2`` enables the image/noise dual-head.
    mlp_ratio : float, optional
        Expansion ratio inside the SwiGLU blocks (default ``1.0``).
    swiglu_mlp : bool, optional
        Retained for constructor compatibility (default ``False``).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        mlp_ratio: float = 1.0,
        swiglu_mlp: bool = False,
    ):
        super().__init__()

        del swiglu_mlp
        token_dim = hidden_size

        # Conditioning-controlled normalization before the two MLP blocks.
        self.norm1 = nn.LayerNorm(token_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(token_dim, elementwise_affine=False, eps=1e-6)

        # Separate output normalizations for image and noise branches.
        self.norm_image = RMSNorm(token_dim, eps=1e-6, elementwise_affine=False)
        self.norm_noise = nn.LayerNorm(token_dim, eps=1e-6, elementwise_affine=False)

        hidden_features = int(token_dim * mlp_ratio)
        self.patch_size = patch_size
        self.input_size = input_size

        self.mlp1 = SwiGLU(
            in_features=token_dim,
            hidden_features=hidden_features,
            norm_layer=nn.LayerNorm,
            drop=0,
        )
        self.mlp2 = SwiGLU(
            in_features=token_dim,
            hidden_features=hidden_features,
            norm_layer=nn.LayerNorm,
            drop=0,
        )

        if out_channels == 2:
            self.linear_nos = nn.Linear(token_dim, patch_size**3 * 1, bias=True)
            self.linear_img = nn.Linear(token_dim, patch_size**3 * 1, bias=True)
        else:
            self.linear_final = nn.Linear(
                token_dim,
                patch_size**3 * out_channels,
                bias=True,
            )

        # Six modulation vectors: two MLP blocks and one dual-head output step.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(token_dim, 6 * token_dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Denoise a patch-token sequence with timestep-conditioned modulation.

        Parameters
        ----------
        x : torch.Tensor
            Patch tokens of shape ``[B, N, token_dim]``.
        c : torch.Tensor
            Coarse-branch timestep conditioning of shape ``[B, token_dim]``.

        Returns
        -------
        torch.Tensor
            For ``out_channels == 2``, dense volume ``[B, 2, D, H, W]`` ordered
            as ``[noise, image]``; otherwise patches ``[B, N, patch_size³ * C]``.
        """
        shift1, scale1, shift2, scale2, shift, scale = self.adaLN_modulation(c).chunk(6, dim=1)

        h = self.mlp1(modulate(self.norm1(x), shift1, scale1))
        h = self.mlp2(modulate(self.norm2(h), shift2, scale2))
        h = h + x

        if hasattr(self, "linear_final"):
            return self.linear_final(modulate(h, shift, scale))

        h_img = self.linear_img(modulate(self.norm_image(h), shift, scale))
        h_noise = self.linear_nos(modulate(self.norm_noise(h), shift, scale))

        img = unpatchify_3d(
            h_img,
            out_channels=1,
            patch_size=self.patch_size,
            input_size=self.input_size,
        )
        noise = unpatchify_3d(
            h_noise,
            out_channels=1,
            patch_size=self.patch_size,
            input_size=self.input_size,
        )
        return torch.cat([noise, img], dim=1)


class FinalLayerRMS(nn.Module):
    """Dual-head final projection layer with RMSNorm on the image branch.

    This is the transformer-side counterpart to `MlpDenoiserRMS`. It projects
    hidden patch tokens into either:

    - separate image/noise predictions when `out_channels == 2`
    - a single shared prediction head otherwise

    If `input_size` is provided, the dual-head outputs are unpatchified back to
    dense 3D volumes before being returned.

    Parameters
    ----------
    hidden_size : int
        Transformer token width entering the final projection.
    patch_size : int
        Edge length of each cubic output patch.
    out_channels : int
        Number of output channels; ``2`` enables the image/noise dual-head.
    input_size : int or None, optional
        Cubic edge length for reconstructing dense output volumes from patches.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, input_size: int = None):
        super().__init__()
        self.norm_noise = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.norm_image = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)

        self.patch_size = patch_size
        self.input_size = input_size
        self.out_channels = out_channels

        if out_channels == 2:
            self.linear_noise = nn.Linear(hidden_size, patch_size**3 * 1, bias=True)
            self.linear_image = nn.Linear(hidden_size, patch_size**3 * 1, bias=True)
        else:
            self.linear = nn.Linear(
                hidden_size,
                patch_size**3 * out_channels,
                bias=True,
            )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Project hidden tokens into patch predictions or dense 3D outputs.

        Parameters
        ----------
        x : torch.Tensor
            Hidden tokens of shape ``[B, N, hidden_size]``.
        c : torch.Tensor
            Conditioning vector of shape ``[B, hidden_size]``.

        Returns
        -------
        torch.Tensor
            Dual-head with ``input_size``: dense ``[B, 2, D, H, W]`` as
            ``[noise, image]``.  Dual-head without ``input_size``: patches
            ``[B, N, 2·patch_size³]``.  Single-head: ``[B, N, C·patch_size³]``.
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)

        if self.out_channels == 2:
            h_noise = self.linear_noise(modulate(self.norm_noise(x), shift, scale))
            h_image = self.linear_image(modulate(self.norm_image(x), shift, scale))

            if self.input_size is not None:
                noise_vol = unpatchify_3d(
                    h_noise,
                    out_channels=1,
                    patch_size=self.patch_size,
                    input_size=self.input_size,
                )
                image_vol = unpatchify_3d(
                    h_image,
                    out_channels=1,
                    patch_size=self.patch_size,
                    input_size=self.input_size,
                )
                return torch.cat([noise_vol, image_vol], dim=1)

            return torch.cat([h_noise, h_image], dim=-1)

        return self.linear(modulate(self.norm_image(x), shift, scale))

class CoarseDenoiserRMS(nn.Module):
    """Coarse denoising path using the RMS dual-head patch MLP variant.

    Extracts raw 3D patches from the input volume and applies
    :class:`MlpDenoiserRMS` to predict separate noise and image outputs.

    Parameters
    ----------
    in_channels : int
        Number of input channels in the volume.
    extract_patch_size : int
        Edge length of the patches extracted from the input.
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
    """
    
    def __init__(self,
                 in_channels: int,
                 extract_patch_size: int,
                 patch_size: int,
                 out_channels: int,
                 input_size: int,
                 stride: int = 4,
                 padding: int = 2,
                 mlp_ratio: float = 1.0,
                 swiglu_mlp: bool = True):
        super().__init__()
        
        # Patch extraction
        self.patch_extractor = ExtractPatches3D(
            patch_size=extract_patch_size,
            stride=stride,
            padding=padding,
        )
        
        # Calculate patch grid dimensions
        self.num_patches, self.grid_size = self.patch_extractor.compute_num_patches(input_size)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.input_size = input_size
        
        # The RMS coarse branch also operates directly in patch space.
        patch_token_dim = in_channels * extract_patch_size**3
        output_volume_size = input_size
        output_patch_size = patch_size
        self.mlp_denoise = MlpDenoiserRMS(
            input_size=output_volume_size,
            hidden_size=patch_token_dim,
            patch_size=output_patch_size,
            out_channels=out_channels,
            swiglu_mlp=swiglu_mlp,
            mlp_ratio=mlp_ratio,
        )
    
    def forward(self, 
                x: torch.Tensor, 
                c: torch.Tensor, 
                return_patches: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the RMS coarse branch on an input volume.

        Parameters
        ----------
        x : torch.Tensor
            Input volume of shape ``[B, C, D, H, W]``.
        c : torch.Tensor
            Coarse-branch timestep conditioning of shape ``[B, patch_token_dim]``.
        return_patches : bool, optional
            Also return the raw extracted patches when ``True`` (default ``False``).

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Coarse output volume, or ``(input_patches, coarse_output)`` when
            ``return_patches=True``.
        """
        # Extract patches from input volume
        patches = self.patch_extractor(x)  # [B, N, C * extract_patch_size^3]

        # Process through MLP denoiser in patch space, then reconstruct volume.
        denoised_patches = self.mlp_denoise(patches, c)
        denoised = unpatchify_3d(
            denoised_patches,
            out_channels=self.out_channels,
            patch_size=self.patch_size,
            input_size=self.input_size,
        )

        return (patches, denoised) if return_patches else denoised
