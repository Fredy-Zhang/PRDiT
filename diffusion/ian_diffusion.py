"""IaN (Image-and-Noise) diffusion process for PRDiT.

Design principles:
1. The model estimates the clean image and noise simultaneously.
2. Forward process: x_t = cos(t/T · π/2) · x_0 + sin(t/T · π/2) · ε
3. Loss:           L = |x_0 - x̂_0|² + |ε - ε̂|²
4. Sampling:       DDIM-style reverse process with an optional predictor-corrector variant.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch


class IaNDiffusion:
    """IaN (Image-and-Noise) predictor-corrector diffusion process.

    The model simultaneously estimates the clean image and the noise at each
    timestep.  The forward process interpolates between them on a cosine
    schedule, and sampling supports both DDIM and predictor-corrector modes.

    Parameters
    ----------
    timestep_respacing : int or None, optional
        Number of timesteps to use during sampling (default: equal to
        ``num_timesteps``).
    loss_type : str, optional
        Reconstruction loss, currently only ``"l2"`` is supported (default
        ``"l2"``).
    num_timesteps : int, optional
        Total number of diffusion steps (default ``1000``).
    """

    def __init__(
        self,
        timestep_respacing: Optional[int] = None,
        loss_type: str = "l2",
        num_timesteps: int = 1000,
    ):
        self.loss_type = loss_type
        self.num_timesteps = num_timesteps
        self.timestep_respacing = int(timestep_respacing) if timestep_respacing else num_timesteps

        # Precompute shared constants.
        self._half_pi = math.pi / 2
        self._step_w = self._half_pi / num_timesteps

    # -- Forward process ------------------------------------------------------

    def gen_noise(self, x_start: torch.Tensor, weight=0.5) -> torch.Tensor:
        """Sample unit Gaussian noise matching the shape of ``x_start``."""
        return torch.randn_like(x_start) * weight

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Diffuse ``x_start`` to timestep ``t``.

        x_t = cos(t/T · π/2) · x_0 + sin(t/T · π/2) · ε
        """
        if noise is None:
            noise = self.gen_noise(x_start)
        t_r = t.reshape(x_start.shape[:1] + (1,) * (x_start.ndim - 1)).to(x_start.device)
        cos_c = torch.cos(t_r / self.num_timesteps * self._half_pi)
        sin_c = torch.sin(t_r / self.num_timesteps * self._half_pi)
        return cos_c * x_start + sin_c * noise

    # -- Training -------------------------------------------------------------

    def training_losses(
        self,
        model: torch.nn.Module,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-sample image and noise reconstruction losses.

        Parameters
        ----------
        model : torch.nn.Module
            The denoising model; must return a tensor splittable into
            ``(eps_recon, img_recon)`` along dim 1.
        x_start : torch.Tensor
            Clean input volumes of shape ``[B, C, D, H, W]``.
        t : torch.Tensor
            Diffusion timesteps of shape ``[B]``.
        model_kwargs : dict or None, optional
            Extra keyword arguments forwarded to the model.

        Returns
        -------
        dict
            ``{"img_loss": Tensor[B], "noise_loss": Tensor[B]}``
        """
        if model_kwargs is None:
            model_kwargs = {}
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)

        noise = self.gen_noise(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        eps_recon, img_recon = model(x_t, t, **model_kwargs).chunk(2, dim=1)

        if self.loss_type != "l2":
            raise NotImplementedError(f"Loss type '{self.loss_type}' is not implemented.")

        spatial_dims = list(range(1, x_start.ndim))
        return {
            "img_loss":   (img_recon - x_start).pow(2).mean(dim=spatial_dims),
            "noise_loss": (eps_recon - noise).pow(2).mean(dim=spatial_dims),
        }

    # -- Sampling -------------------------------------------------------------

    def p_sample_loop(
        self,
        model: torch.nn.Module,
        shape: Tuple[int, ...],
        z: torch.Tensor,
        clip_denoised: bool = False,
        progress: bool = False,
        new_sampling: bool = False,
        device: Optional[torch.device] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Run full reverse diffusion from noise ``z``.

        Parameters
        ----------
        model : torch.nn.Module
            The denoising model.
        shape : tuple of int
            Shape of the output tensor (unused when ``z`` is provided).
        z : torch.Tensor
            Initial noise tensor.
        clip_denoised : bool, optional
            Unused; kept for API compatibility (default ``False``).
        progress : bool, optional
            Unused; kept for API compatibility (default ``False``).
        new_sampling : bool, optional
            When ``True``, use the predictor-corrector sampler; otherwise use
            the standard DDIM sampler (default ``False``).
        device : torch.device or None, optional
            Unused; kept for API compatibility (default ``None``).
        model_kwargs : dict or None, optional
            Extra keyword arguments forwarded to the model.

        Returns
        -------
        tuple
            ``(xs, x0_final)`` — *xs* is the list of trajectory tensors and
            *x0_final* is the last predicted clean image.
        """
        if model_kwargs is None:
            model_kwargs = {}

        skip = self.num_timesteps // self.timestep_respacing
        seq = range(0, self.num_timesteps, skip)

        if new_sampling:
            print("Using the predictor-corrector sampling approach.")
            x0_preds, xs = self._predictor_corrector_steps(z, seq, model, model_kwargs)
        else:
            print("Using the standard DDIM sampling approach.")
            x0_preds, xs = self._generalized_steps(z, seq, model, model_kwargs)

        return xs, x0_preds[-1]

    @torch.no_grad()
    def _generalized_steps(
        self,
        x: torch.Tensor,
        seq: range,
        model: torch.nn.Module,
        model_kwargs: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Standard DDIM-style reverse diffusion."""
        if model_kwargs is None:
            model_kwargs = {}

        n = x.size(0)
        seq_next = [0] + list(seq[:-1])
        x0_preds, xs = [], [x]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t      = torch.full((n,), i, device=x.device)
            next_t = torch.full((n,), j, device=x.device)
            at      = (1 - (t      / self.num_timesteps)[:, None, None, None, None])
            next_at = (1 - (next_t / self.num_timesteps)[:, None, None, None, None])

            xt = xs[-1].to(x.device)
            eps_recon, img_recon = model(xt, t, **model_kwargs).chunk(2, dim=1)

            xt_next = xt - (at - next_at) * self._half_pi * (
                torch.cos(at * self._half_pi) * img_recon
                - torch.sin(at * self._half_pi) * eps_recon
            )

            x0_preds.append(img_recon.cpu())
            xs.append(xt_next.cpu())

        return x0_preds, xs

    @torch.no_grad()
    def _predictor_corrector_steps(
        self,
        x: torch.Tensor,
        seq: range,
        model: torch.nn.Module,
        model_kwargs: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Predictor-corrector sampler with stochastic noise injection.

        Uses a 2-step predictor jump followed by an exact corrector that
        re-injects noise to land at the target timestep.
        """
        if model_kwargs is None:
            model_kwargs = {}

        p = 2
        n = x.size(0)
        seq_next = [0] + list(seq[:-1])
        xs, x0_preds = [x], []
        device = x.device

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((n,), i, device=device)
            h = (j - i) * self._step_w  # negative step size

            xt = xs[-1].to(device)
            et, x0_t = model(xt, t, **model_kwargs).chunk(2, dim=1)

            beta_t = (t / self.num_timesteps)[..., None, None, None, None] * self._half_pi
            f_t = torch.sin(beta_t) * x0_t - torch.cos(beta_t) * et

            if i - p * (i - j) > 0:
                # Predictor: jump back p steps.
                xt_pred = xt - p * h * f_t
                t_pred = torch.full((n,), i - p * (i - j), device=device)
                t_corr = torch.full((n,), i - (i - j),     device=device)

                beta_pred = (t_pred / self.num_timesteps)[..., None, None, None, None] * self._half_pi
                beta_corr = (t_corr / self.num_timesteps)[..., None, None, None, None] * self._half_pi

                alpha   = torch.cos(beta_corr) / torch.cos(beta_pred)
                alpha_1 = torch.sqrt(torch.clamp(1 - alpha ** 2, min=0.0))

                # Corrector: stochastic re-injection to land at t_corr.
                xt_next = alpha * xt_pred + alpha_1 * self.gen_noise(xt_pred)
            else:
                xt_next = xt - h * f_t

            x0_preds.append(x0_t.cpu())
            xs.append(xt_next.cpu())

        return x0_preds, xs
