"""Stage-3 super-resolution refinement for PRDiT.

Stage 1 (`models.local_denoiser.CoarseDenoiser`) and stage 2
(`models.global_refiner.FineRefiner`), orchestrated by `models.models.PRDiT`,
train a diffusion model at a single resolution. This module adds a stage-3
wrapper, `PRDiTSR`, that refines a trained stage-1+2 `PRDiT` model up to a
higher target resolution:

1. downsample the noisy full-resolution input (nearest-neighbor)
2. run it through the (frozen or jointly fine-tunable) base `PRDiT`
3. upsample the base prediction back to full resolution — trilinear for the
   image channel, nearest-neighbor for the noise channel
4. add a full-resolution local correction predicted by a `CoarseDenoiser`
   instance sized for the target grid

This mirrors the "High_DiT" design validated in the `3-2-nearest` experiment,
including its defining choice versus the earlier "3-2" variant: nearest
(not custom std-matched) noise upsampling, and gradients allowed to reach the
base model whenever it is set to fine-tune.
"""

from __future__ import annotations

import logging
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.local_denoiser import CoarseDenoiser
from models.models import PRDiT, TimestepEmbedder
from models.utils import unpatchify_3d
from util import requires_grad


logger = logging.getLogger(__name__)


class PRDiTSR(nn.Module):
    """Stage-3 super-resolution refiner wrapping a base stage-1+2 PRDiT model.

    Parameters
    ----------
    base : PRDiT
        Trained stage-1+2 model, constructed at ``base_input_size``.
    base_input_size : int
        Spatial edge length the base model operates at (e.g. ``128``).
    target_input_size : int
        Spatial edge length of the super-resolved output (e.g. ``256``).
    in_channels : int, optional
        Number of input voxel channels (default ``1``).
    sr_hidden_size : int, optional
        Patch-token width for the SR correction head (default ``768``).
    sr_patch_size : int, optional
        Patch-extraction edge length for the SR correction head (default ``12``).
    sr_stride : int, optional
        Patch-extraction stride, and output patch edge length, for the SR
        correction head (default ``8``).
    sr_padding : int, optional
        Reflection padding before patch extraction (default ``2``).
    base_finetune : bool, optional
        When ``True``, gradients flow into ``base`` and its parameters are
        trainable; when ``False`` (default), ``base`` is frozen and run under
        ``torch.no_grad()`` to save memory.
    """

    def __init__(
        self,
        base: PRDiT,
        base_input_size: int,
        target_input_size: int,
        in_channels: int = 1,
        sr_hidden_size: int = 768,
        sr_patch_size: int = 12,
        sr_stride: int = 8,
        sr_padding: int = 2,
        base_finetune: bool = False,
    ):
        super().__init__()
        self.base = base
        self.base_input_size = base_input_size
        self.target_input_size = target_input_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2  # eps + img, concatenated

        # Single-head timestep embedder conditioning the SR correction MLP,
        # sized like PRDiT's own coarse conditioning head.
        self.t_embedder_sr = TimestepEmbedder(
            hidden_size=sr_hidden_size,
            coarse_hidden_size=int(in_channels * sr_patch_size**3),
            fine_hidden_size=sr_hidden_size,
            frequency_embedding_size=256,
            is_depth_zero=True,
        )

        # The SR correction is a local per-patch MLP denoiser (same family as
        # stage-1's CoarseDenoiser), not a global-attention transformer:
        # doubling resolution 128->256 gives 8x more patches, so a per-patch
        # MLP scales linearly where global attention would scale quadratically.
        self.sr = CoarseDenoiser(
            in_channels=in_channels,
            extract_patch_size=sr_patch_size,
            hidden_size=sr_hidden_size,
            patch_size=sr_stride,
            out_channels=self.out_channels,
            input_size=target_input_size,
            stride=sr_stride,
            padding=sr_padding,
            mlp_ratio=1.0,
            swiglu_mlp=True,
        )
        self._sr_output_patch_size = sr_stride

        self.set_base_finetune(base_finetune)
        self.initialize_weights()

    def set_base_finetune(self, flag: bool) -> None:
        """Toggle whether the wrapped base PRDiT is jointly fine-tuned."""
        self._base_finetune = flag
        requires_grad(self.base, flag)

    def initialize_weights(self) -> None:
        """Zero-init the SR head's output projection.

        Matches PRDiT's own coarse-path init: the correction starts as a
        no-op on top of the upsampled base prediction.
        """
        nn.init.constant_(self.sr.mlp_denoise.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.sr.mlp_denoise.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.sr.mlp_denoise.linear_final.weight, 0)
        nn.init.constant_(self.sr.mlp_denoise.linear_final.bias, 0)

    def load_base_checkpoint(self, checkpoint_path: str, map_location: str = "cpu") -> None:
        """Load a trained stage-2 PRDiT checkpoint into ``self.base``."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.base.load_state_dict(state_dict, strict=True)
        logger.info("Loaded base PRDiT checkpoint from %s", checkpoint_path)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run stage-3 super-resolution refinement.

        Returns a ``[B, 2*in_channels, D, H, W]`` tensor (eps, img concatenated
        along the channel dim) by default, matching `PRDiT.forward`'s output
        contract — so this model is a drop-in ``model`` argument for
        ``diffusion.training_losses``/``train_step``/``p_sample_loop`` with no
        changes to the `diffusion` package.

        When ``return_intermediate=True``, returns
        ``(base_upsampled, local_correction)`` as two separately-summed
        ``[B, 2*in_channels, D, H, W]`` tensors instead.
        """
        grad_ctx = torch.enable_grad() if self._base_finetune else torch.no_grad()
        with grad_ctx:
            x_lo = F.interpolate(x, size=(self.base_input_size,) * 3, mode="nearest")
            base_out = self.base(x_lo, t)
            eps_recon, img_recon = base_out.chunk(2, dim=1)

        img_high = F.interpolate(img_recon, size=x.shape[2:], mode="trilinear", align_corners=False)
        # Nearest (not std-matched) noise upsampling — the defining choice of
        # the "-nearest" variant this stage is ported from.
        noise_high = F.interpolate(eps_recon, size=x.shape[2:], mode="nearest")

        c_sr, _ = self.t_embedder_sr(t)
        local_patches = self.sr(x, c_sr)
        local_pred = unpatchify_3d(
            local_patches,
            out_channels=self.out_channels,
            patch_size=self._sr_output_patch_size,
            input_size=self.target_input_size,
        )
        local_eps, local_img = local_pred.chunk(2, dim=1)

        if return_intermediate:
            return (
                torch.cat([noise_high, img_high], dim=1),
                torch.cat([local_eps, local_img], dim=1),
            )
        return torch.cat([local_eps + noise_high, local_img + img_high], dim=1)


def load_sr_model(config) -> PRDiTSR:
    """Construct a `PRDiTSR` from an experiment config.

    Builds the base stage-1+2 model from ``config.base_model.*`` at
    ``config.data.base_image_size``, builds the SR correction head from
    ``config.sr_model.*`` at ``config.data.image_size``, loads
    ``config.base_model.checkpoint`` into the base model, and applies
    ``config.base_model.finetune``.
    """
    from models import PRDiT_models  # deferred: avoids a package-init cycle

    base_name = config.base_model.name
    if base_name not in PRDiT_models:
        raise ValueError(f"Base model name {base_name} is not recognized.")

    base = PRDiT_models[base_name](
        input_size=config.data.base_image_size,
        in_channels=config.base_model.in_channels,
        num_classes=getattr(config.base_model, "num_classes", 1),
        learn_sigma=(config.base_model.out_channels == 2),
        flash_attn=getattr(config.base_model, "flash_attn", True),
    )

    model = PRDiTSR(
        base=base,
        base_input_size=config.data.base_image_size,
        target_input_size=config.data.image_size,
        in_channels=config.base_model.in_channels,
        sr_hidden_size=config.sr_model.hidden_size,
        sr_patch_size=config.sr_model.patch_size,
        sr_stride=config.sr_model.stride,
        sr_padding=config.sr_model.padding,
        base_finetune=bool(getattr(config.base_model, "finetune", False)),
    )

    checkpoint_path = getattr(config.base_model, "checkpoint", None)
    if checkpoint_path:
        model.load_base_checkpoint(checkpoint_path)

    return model
