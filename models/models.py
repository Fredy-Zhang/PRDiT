"""Top-level PRDiT model orchestration.

This module wires together the shared timestep-conditioning path with:

- stage 1: the local coarse denoiser in `models.local_denoiser`
- stage 2: the global residual DiT refiner in `models.global_refiner`

Keeping the high-level `PRDiT` class here makes the training flow easy to
follow, while the stage-specific implementation details live in their own
modules.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from models.global_refiner import FineRefiner
from models.local_denoiser import CoarseDenoiser
from util import requires_grad


logger = logging.getLogger(__name__)


class TimestepEmbedder(nn.Module):
    """
    Timestep embedder with separate coarse and fine conditioning heads.

    Scalar diffusion timesteps are first converted into sinusoidal embeddings,
    then passed through a shared MLP. The resulting shared representation is
    projected into two different conditioning spaces:

    - a coarse embedding for the MLP denoiser
    - a fine embedding for the transformer refiner

    This split is important for staged training because the coarse branch can be
    frozen while the fine branch keeps learning.
    """

    def __init__(
        self,
        hidden_size: int,
        coarse_hidden_size: int,
        fine_hidden_size: int,
        frequency_embedding_size: int = 256,
        is_depth_zero: bool = True,
    ):
        """Initialize the timestep embedder.

        Parameters
        ----------
        hidden_size : int
            Width of the shared MLP hidden layer.
        coarse_hidden_size : int
            Output dimension for the coarse conditioning head.
        fine_hidden_size : int
            Output dimension for the fine conditioning head; ignored when
            ``is_depth_zero=True`` (head becomes ``nn.Identity``).
        frequency_embedding_size : int, optional
            Dimensionality of the sinusoidal positional embedding
            (default ``256``).
        is_depth_zero : bool, optional
            When ``True`` the fine head is replaced with an identity map so
            that only stage-1 parameters are created (default ``True``).
        """
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
        )

        self.coarse_head = nn.Linear(hidden_size, coarse_hidden_size, bias=True)
        self.fine_head = (
            nn.Identity()
            if is_depth_zero
            else nn.Linear(hidden_size, fine_hidden_size, bias=True)
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process timesteps through the shared MLP and both branch heads."""
        timestep_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        shared_features = self.mlp(timestep_emb)
        coarse_emb = self.coarse_head(shared_features)
        fine_emb = self.fine_head(shared_features)
        return coarse_emb, fine_emb


class PRDiT(nn.Module):
    """
    Main PRDiT architecture for 3D diffusion-based volume modeling.

    PRDiT combines two complementary branches:

    - a stage-1 coarse MLP denoiser that works directly on extracted 3D patches
    - a stage-2 fine transformer refiner that learns residual corrections on top
      of the coarse prediction
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        stride: int = 4,
        padding: int = 2,
        in_channels: int = 1,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1,
        learn_sigma: bool = False,
        flash_attn: bool = False,
    ):
        """Initialize the PRDiT model.

        Parameters
        ----------
        input_size : int, optional
            Spatial edge length of input volumes (default ``32``).
        patch_size : int, optional
            Patch edge length used by the coarse extractor (default ``2``).
        stride : int, optional
            Output patch stride used for un-patchification (default ``4``).
        padding : int, optional
            Padding applied in the patch-extraction convolution (default ``2``).
        in_channels : int, optional
            Number of input voxel channels (default ``1``).
        hidden_size : int, optional
            Shared transformer hidden dimension (default ``1152``).
        depth : int, optional
            Number of transformer blocks in the stage-2 refiner; ``0`` for
            stage-1-only mode (default ``28``).
        num_heads : int, optional
            Attention heads in each transformer block (default ``16``).
        mlp_ratio : float, optional
            MLP expansion factor inside transformer blocks (default ``4.0``).
        class_dropout_prob : float, optional
            Unused — kept for API compatibility (default ``0.1``).
        num_classes : int, optional
            Unused — kept for API compatibility (default ``1``).
        learn_sigma : bool, optional
            When ``True`` the output channels are doubled to predict both the
            mean and variance (default ``False``).
        flash_attn : bool, optional
            Enable FlashAttention in stage-2 transformer blocks (default
            ``False``).
        """
        super().__init__()
        del class_dropout_prob, num_classes

        extract_patch_size = patch_size
        output_patch_size = stride

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size = input_size
        self.extract_patch_size = extract_patch_size
        self.patch_size = output_patch_size
        self.depth = depth
        self.hidden_size = hidden_size

        self._config_to_log = {
            "input_size": input_size,
            "extract_patch_size": extract_patch_size,
            "output_patch_size": output_patch_size,
            "stride": stride,
            "in_channels": in_channels,
            "hidden_size": hidden_size,
            "depth": depth,
            "num_heads": num_heads,
            "learn_sigma": learn_sigma,
        }

        self.t_embedder = TimestepEmbedder(
            hidden_size=hidden_size,
            coarse_hidden_size=int(in_channels * extract_patch_size**3),
            fine_hidden_size=hidden_size,
            frequency_embedding_size=256,
            is_depth_zero=(depth == 0),
        )

        self.coarse = CoarseDenoiser(
            in_channels=in_channels,
            hidden_size=hidden_size,
            extract_patch_size=extract_patch_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
            input_size=input_size,
            stride=stride,
            padding=padding,
            mlp_ratio=1.0,
            swiglu_mlp=True,
        )

        self.fine = None
        if depth > 0:
            self.fine = FineRefiner(
                in_channels=in_channels,
                extract_patch_size=extract_patch_size,
                hidden_size=hidden_size,
                patch_size=self.patch_size,
                out_channels=self.out_channels,
                depth=depth,
                num_heads=num_heads,
                num_patches=self.coarse.num_patches,
                input_size=input_size,
                stride=stride,
                padding=padding,
                mlp_ratio=mlp_ratio,
                flash_attn=flash_attn,
            )

        self.initialize_weights()

        if depth > 0:
            self.freeze_coarse_path()
            logger.info("Stage 2 setup: Coarse path frozen, training %s transformer layers", depth)

    def log_config(self, rank: int = 0) -> None:
        """Log the stored model configuration from the primary process."""
        if rank == 0 and hasattr(self, "_config_to_log"):
            logger.info("PRDiT Model Configuration:")
            for key, value in self._config_to_log.items():
                logger.info("  %s: %s", key, value)

    def _log_config(self, config: dict, rank: int = 0) -> None:
        """Log an explicit configuration dictionary from the primary process."""
        if rank == 0:
            logger.info("PRDiT Model Configuration:")
            for key, value in config.items():
                logger.info("  %s: %s", key, value)

    def freeze_coarse_path(self) -> None:
        """Freeze the stage-1 coarse branch and its timestep-conditioning path."""
        requires_grad(self.coarse, False)
        requires_grad(self.t_embedder.coarse_head, False)
        requires_grad(self.t_embedder.mlp, False)

        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Frozen %s parameters, %s trainable", f"{frozen_params:,}", f"{trainable_params:,}")

    def initialize_weights(self, gain: float = 1.0) -> None:
        """Initialize PRDiT weights with branch-specific schemes."""
        logger.info("Initializing model weights...")

        def _init_linear_layers(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv3d):
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_init_linear_layers)
        self._init_timestep_embedder()
        self._init_coarse_path()

        if self.depth > 0 and self.fine is not None:
            self._init_fine_path(gain)

        logger.info("Weight initialization complete")

    def _init_timestep_embedder(self) -> None:
        """Initialize the shared timestep MLP and its coarse/fine output heads."""
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        if self.t_embedder.mlp[0].bias is not None:
            nn.init.zeros_(self.t_embedder.mlp[0].bias)

        nn.init.normal_(self.t_embedder.coarse_head.weight, std=0.02)
        if self.t_embedder.coarse_head.bias is not None:
            nn.init.zeros_(self.t_embedder.coarse_head.bias)

        if not isinstance(self.t_embedder.fine_head, nn.Identity):
            nn.init.normal_(self.t_embedder.fine_head.weight, std=0.02)
            if self.t_embedder.fine_head.bias is not None:
                nn.init.zeros_(self.t_embedder.fine_head.bias)

    def _init_coarse_path(self) -> None:
        """Initialize the stage-1 coarse branch from a stable zero-output regime."""
        nn.init.constant_(self.coarse.mlp_denoise.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.coarse.mlp_denoise.adaLN_modulation[-1].bias, 0)

        if hasattr(self.coarse.mlp_denoise, "linear_final"):
            nn.init.constant_(self.coarse.mlp_denoise.linear_final.weight, 0)
            nn.init.constant_(self.coarse.mlp_denoise.linear_final.bias, 0)
        elif hasattr(self.coarse.mlp_denoise, "linear_img"):
            nn.init.constant_(self.coarse.mlp_denoise.linear_img.weight, 0)
            nn.init.constant_(self.coarse.mlp_denoise.linear_img.bias, 0)
            if hasattr(self.coarse.mlp_denoise, "linear_nos"):
                nn.init.constant_(self.coarse.mlp_denoise.linear_nos.weight, 0)
                nn.init.constant_(self.coarse.mlp_denoise.linear_nos.bias, 0)

    def _init_fine_path(self, gain: float) -> None:
        """Initialize the stage-2 transformer refinement branch and its output head."""
        if hasattr(self.fine, "patch_embedder"):
            nn.init.xavier_uniform_(self.fine.patch_embedder.fc1.weight, gain=gain)
            nn.init.xavier_uniform_(self.fine.patch_embedder.fc2.weight, gain=gain)
            nn.init.xavier_uniform_(self.fine.patch_embedder.skip.weight, gain=0.1)

            for layer in [self.fine.patch_embedder.fc1, self.fine.patch_embedder.fc2]:
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        for block in self.fine.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.fine.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.fine.final_layer.adaLN_modulation[-1].bias, 0)

        if hasattr(self.fine.final_layer, "linear"):
            nn.init.constant_(self.fine.final_layer.linear.weight, 0)
            nn.init.constant_(self.fine.final_layer.linear.bias, 0)
        elif hasattr(self.fine.final_layer, "linear_image"):
            nn.init.constant_(self.fine.final_layer.linear_image.weight, 0)
            nn.init.constant_(self.fine.final_layer.linear_image.bias, 0)
            if hasattr(self.fine.final_layer, "linear_noise"):
                nn.init.constant_(self.fine.final_layer.linear_noise.weight, 0)
                nn.init.constant_(self.fine.final_layer.linear_noise.bias, 0)

    def unpatchify_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct a dense 3D volume from patch-space predictions."""
        channels = self.out_channels
        patch_edge = self.patch_size
        grid_size = self.input_size // patch_edge

        x = x.reshape(-1, grid_size, grid_size, grid_size, patch_edge, patch_edge, patch_edge, channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(-1, channels, grid_size * patch_edge, grid_size * patch_edge, grid_size * patch_edge)

    def forward(
        self,
        input: torch.Tensor,
        t: Optional[Union[torch.Tensor, int]] = None,
        y: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
        mode: str = "denoise",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run PRDiT in denoising mode or encoder-feature mode."""
        del y
        if mode not in {"denoise", "encode"}:
            raise ValueError(f"Invalid mode '{mode}'. Supported modes are ['denoise', 'encode'].")

        if t is None:
            if mode == "encode":
                t = torch.full((input.shape[0],), 50, device=input.device, dtype=torch.long)
            else:
                raise ValueError("Denoise mode requires an explicit timestep tensor `t`.")
        elif isinstance(t, int):
            t = torch.full((input.shape[0],), t, device=input.device, dtype=torch.long)
        elif t.ndim == 0:
            t = t.expand(input.shape[0]).to(device=input.device)
        else:
            t = t.to(device=input.device)

        c_coarse, c_fine = self.t_embedder(t)

        if mode == "encode":
            if self.fine is None:
                raise RuntimeError("Encode mode requires transformer depth > 0 (self.fine is None).")
            if len(self.fine.blocks) <= 5:
                raise RuntimeError(
                    f"Encode mode requires at least 6 transformer blocks; found {len(self.fine.blocks)}."
                )

            patches = self.coarse.patch_extractor(input)
            _, features = self.fine.forward_hidden_features(
                patches,
                c_fine,
                capture_layers=5,
                include_patch_embed=False,
            )
            return features["block_5"]

        with torch.no_grad() if self.depth > 0 else torch.enable_grad():
            if self.depth > 0 or return_intermediate:
                patches, coarse_out = self.coarse(input, c_coarse, return_patches=True)
            else:
                coarse_out = self.coarse(input, c_coarse)

        if self.depth > 0 and self.fine is not None:
            fine_out = self.fine(patches, c_fine)

            if return_intermediate:
                return self.unpatchify_3d(coarse_out), self.unpatchify_3d(fine_out)

            x = coarse_out + fine_out
        else:
            x = coarse_out

        x = self.unpatchify_3d(x)
        return x

    def get_vlm_features(
        self,
        x: torch.Tensor,
        timestep: int = 50,
        no_grad: bool = True,
        detach: bool = False,
    ) -> torch.Tensor:
        """Return token features from the 6th transformer block for VLM integration."""
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            features = self.forward(x, t=timestep, mode="encode")
        if detach:
            features = features.detach()
        return features

    def extract_transformer_hidden_features(
        self,
        input: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        capture_layers: Optional[Union[int, Sequence[int]]] = None,
        include_patch_embed: bool = False,
        detach: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Extract hidden token features from the stage-2 transformer branch."""
        del y

        if self.fine is None:
            raise RuntimeError("Transformer hidden features are only available when PRDiT depth > 0.")

        _, c_fine = self.t_embedder(t)
        patches = self.coarse.patch_extractor(input)
        _, features = self.fine.forward_hidden_features(
            patches,
            c_fine,
            capture_layers=capture_layers,
            include_patch_embed=include_patch_embed,
        )

        if detach:
            return {name: value.detach() for name, value in features.items()}
        return features

    def load_coarse_checkpoint(self, checkpoint_path: str) -> None:
        """Load a stage-1 checkpoint and use it to initialize the coarse branch."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(checkpoint["model"], strict=False)
        self.freeze_coarse_path()
        logger.info("Loaded stage 1 checkpoint from %s and froze coarse path", checkpoint_path)
