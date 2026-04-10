"""Global residual DiT components for PRDiT.

This module groups the transformer-based refinement path used in stage-2:

- `PatchEmbed3D`: embed patch tokens into the transformer space
- `PRDiTBlock`: conditioned transformer refinement block
- `FinalLayer`: map transformer tokens back to patch predictions
- `FineRefiner`: end-to-end stage-2 residual refinement branch
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp

from models.classes import Attention
from models.utils import get_normalized_3d_pos_enc, modulate, get_3d_sincos_pos_embed


logger = logging.getLogger(__name__)


class PatchEmbed3D(nn.Module):
    """Embed flattened 3D patches with an MLP plus a linear skip projection.

    The main path learns a nonlinear projection into the hidden space, while the
    skip path preserves a direct linear route from raw patch values to the final
    embedding. This makes the patch projection more expressive without losing a
    simple residual path.
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = nn.LayerNorm,
        mlp_ratio: float = 4.0,
        activation: Callable = nn.GELU(approximate="tanh"),
        dropout: float = 0.0,
    ):
        """Initialize the 3D patch embedder.

        Parameters
        ----------
        patch_size : int, optional
            Cubic patch edge length; the raw token dimension is
            ``in_chans * patch_size³`` (default ``16``).
        in_chans : int, optional
            Number of input volume channels (default ``1``).
        embed_dim : int, optional
            Output embedding dimension (default ``768``).
        norm_layer : callable or None, optional
            Normalization applied after the residual sum; ``None`` disables it
            (default ``nn.LayerNorm``).
        mlp_ratio : float, optional
            Hidden-layer expansion factor relative to ``embed_dim``
            (default ``4.0``).
        activation : callable, optional
            Nonlinearity applied between the two MLP layers
            (default ``nn.GELU(approximate='tanh')``).
        dropout : float, optional
            Dropout probability applied to the output embedding (default ``0.0``).
        """
        super().__init__()

        input_dim = in_chans * (patch_size**3)
        hidden_dim = int(embed_dim * mlp_ratio)

        logger.debug(
            "PatchEmbed3D: input_dim=%s, hidden_dim=%s, embed_dim=%s",
            input_dim,
            hidden_dim,
            embed_dim,
        )

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.act = activation
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=True)
        self.skip = nn.Linear(input_dim, embed_dim, bias=False)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed patch tokens into the transformer hidden space.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, N, in_chans * patch_size³)``
            Flattened 3D patch tokens.

        Returns
        -------
        torch.Tensor, shape ``(B, N, embed_dim)``
            Normalized and optionally dropped patch embeddings.
        """
        h = self.fc1(x)
        h = self.act(h)
        h = self.fc2(h)
        s = self.skip(x)
        out = self.norm(h + s)
        return self.drop(out)


class DiTBlock(nn.Module):
    """Transformer refinement block with AdaLN-Zero conditioning.

    Each block applies attention and MLP updates under timestep-conditioned
    modulation. The conditioning network predicts the shift, scale, and gating
    values used to control both sublayers.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        flash_attn: bool = False,
        **block_kwargs,
    ):
        """Initialize a DiT transformer block.

        Parameters
        ----------
        hidden_size : int
            Token feature dimension.
        num_heads : int
            Number of self-attention heads.
        mlp_ratio : float, optional
            MLP hidden-layer expansion factor (default ``4.0``).
        flash_attn : bool, optional
            Enable ``F.scaled_dot_product_attention`` in the attention layer
            (default ``False``).
        **block_kwargs
            Extra keyword arguments forwarded to :class:`~models.classes.Attention`.
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            use_flash_attention=flash_attn,
            **block_kwargs,
        )

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply one transformer refinement block under timestep conditioning.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, N, hidden_size)``
            Input token sequence.
        c : torch.Tensor, shape ``(B, hidden_size)``
            Timestep conditioning vector.

        Returns
        -------
        torch.Tensor, shape ``(B, N, hidden_size)``
            Updated token sequence after attention and MLP sublayers.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final processing layer for the transformer refinement branch.

    Maps transformer token features back into patch-space outputs under
    timestep conditioning.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        """Initialize the final projection layer.

        Parameters
        ----------
        hidden_size : int
            Transformer token width entering this layer.
        patch_size : int
            Cubic output patch edge length; output width is
            ``patch_size³ * out_channels``.
        out_channels : int
            Number of output channels predicted per patch voxel.
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size**3 * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Project transformer tokens back to patch predictions.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, N, hidden_size)``
            Token sequence from the last transformer block.
        c : torch.Tensor, shape ``(B, hidden_size)``
            Timestep conditioning vector.

        Returns
        -------
        torch.Tensor, shape ``(B, N, patch_size³ * out_channels)``
            Patch-space output predictions.
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class FineRefiner(nn.Module):
    """Transformer-based residual refinement branch of PRDiT.

    Takes patch tokens from the coarse branch, adds sinusoidal positional
    encodings, and refines the representation with :class:`DiTBlock` attention
    blocks. Models higher-frequency residual detail that the coarse MLP misses.
    """

    def __init__(
        self,
        in_channels: int,
        extract_patch_size: int,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        depth: int,
        num_heads: int,
        num_patches: int,
        input_size: int,
        stride: int = 4,
        padding: int = 2,
        mlp_ratio: float = 4.0,
        flash_attn: bool = False,
    ):
        """Initialize the stage-2 fine refinement branch.

        Parameters
        ----------
        in_channels : int
            Number of input volume channels.
        extract_patch_size : int
            Edge length of patches produced by the coarse extractor.
        hidden_size : int
            Transformer hidden dimension.
        patch_size : int
            Output patch edge length used to compute the positional-encoding
            grid (``grid_size = input_size // patch_size``).
        out_channels : int
            Number of output channels predicted per patch voxel.
        depth : int
            Number of :class:`DiTBlock` transformer layers.
        num_heads : int
            Number of self-attention heads per block.
        num_patches : int
            Unused; retained for API compatibility with the model registry.
        input_size : int
            Cubic edge length of the input/output volume.
        stride : int, optional
            Unused; retained for API compatibility (default ``4``).
        padding : int, optional
            Unused; retained for API compatibility (default ``2``).
        mlp_ratio : float, optional
            MLP hidden-layer expansion factor in each block (default ``4.0``).
        flash_attn : bool, optional
            Enable FlashAttention in every :class:`DiTBlock` (default ``False``).
        """
        super().__init__()
        del num_patches, stride, padding

        logger.debug("FineRefiner: depth=%s, hidden_size=%s, num_heads=%s", depth, hidden_size, num_heads)

        self.patch_embedder = PatchEmbed3D(
            patch_size=extract_patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            norm_layer=nn.LayerNorm,
            activation=nn.GELU(approximate="tanh"),
        )

        grid_size = input_size // patch_size
        pos_embed = get_normalized_3d_pos_enc(grid_size=grid_size, embed_dim=hidden_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    flash_attn=flash_attn,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        self.input_size = input_size
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Refine patch tokens with a global residual transformer.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, N, in_chans * extract_patch_size³)``
            Raw patch tokens from the coarse branch.
        c : torch.Tensor, shape ``(B, hidden_size)``
            Timestep conditioning vector.

        Returns
        -------
        torch.Tensor, shape ``(B, N, patch_size³ * out_channels)``
            Residual patch predictions from the final projection layer.
        """
        h = self.patch_embedder(x)
        h = h + self.pos_embed

        for block in self.blocks:
            h = block(h, c)

        return self.final_layer(h, c)

    def get_recommended_capture_layer(self) -> int:
        """Return the default transformer block index for encoder-style features."""
        if len(self.blocks) == 0:
            raise RuntimeError("No transformer blocks are available for hidden-feature extraction.")
        return len(self.blocks) // 2

    def _normalize_capture_layers(
        self,
        capture_layers: Optional[Union[int, Sequence[int]]],
    ) -> set[int]:
        """Normalize and validate requested encoder feature layer indices.

        Parameters
        ----------
        capture_layers : int, sequence of int, or None
            Block indices to capture. ``None`` defaults to the middle block.

        Returns
        -------
        set of int
            Validated set of block indices in ``[0, depth)``.

        Raises
        ------
        ValueError
            If any index falls outside the valid range.
        """
        if capture_layers is None:
            return {self.get_recommended_capture_layer()}

        if isinstance(capture_layers, int):
            normalized = {capture_layers}
        else:
            normalized = {int(layer_idx) for layer_idx in capture_layers}

        invalid = [layer_idx for layer_idx in sorted(normalized) if layer_idx < 0 or layer_idx >= len(self.blocks)]
        if invalid:
            raise ValueError(
                f"capture_layers contains invalid layer indices {invalid}; "
                f"valid range is [0, {len(self.blocks) - 1}]"
            )
        return normalized

    def forward_hidden_features(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        capture_layers: Optional[Union[int, Sequence[int]]] = None,
        include_patch_embed: bool = False,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the transformer branch and capture hidden token features.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, N, patch_dim)``
            Raw patch tokens from the coarse branch.
        c : torch.Tensor, shape ``(B, hidden_size)``
            Timestep conditioning vector.
        capture_layers : int, sequence of int, or None, optional
            Block indices whose outputs are stored in the returned dict.
            ``None`` defaults to the middle block (default ``None``).
        include_patch_embed : bool, optional
            When ``True``, also store the post-embedding tokens under the
            ``"patch_embed"`` key (default ``False``).

        Returns
        -------
        h : torch.Tensor, shape ``(B, N, hidden_size)``
            Final hidden-state after all blocks.
        features : dict[str, torch.Tensor]
            Captured intermediate states keyed as ``"block_i"``,
            optionally ``"patch_embed"``, and always ``"final"``.
        """
        requested_layers = self._normalize_capture_layers(capture_layers)
        features: dict[str, torch.Tensor] = {}

        h = self.patch_embedder(x)
        h = h + self.pos_embed
        if include_patch_embed:
            features["patch_embed"] = h

        for layer_idx, block in enumerate(self.blocks):
            h = block(h, c)
            if layer_idx in requested_layers:
                features[f"block_{layer_idx}"] = h

        features["final"] = h
        return h, features

    def get_layer_features(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        capture_layers: Optional[Union[int, Sequence[int]]] = None,
        include_patch_embed: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return selected transformer hidden states for encoder-style reuse.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, N, patch_dim)``
            Raw patch tokens from the coarse branch.
        c : torch.Tensor, shape ``(B, hidden_size)``
            Timestep conditioning vector.
        capture_layers : int, sequence of int, or None, optional
            Block indices to return (default: middle block).
        include_patch_embed : bool, optional
            Also include the ``"patch_embed"`` key (default ``False``).

        Returns
        -------
        dict[str, torch.Tensor]
            Captured hidden states; see :meth:`forward_hidden_features`.
        """
        _, features = self.forward_hidden_features(
            x,
            c,
            capture_layers=capture_layers,
            include_patch_embed=include_patch_embed,
        )
        return features
