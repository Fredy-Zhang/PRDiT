"""
Image and Noise Diffusion (IaN) Module

A diffusion approach that simultaneously estimates image and noise components.
The model uses trigonometric interpolation between image and noise:
- xt = image * cos(t/T * pi/2) + noise * sin(t/T * pi/2)
- Loss combines image reconstruction and noise estimation
- Sampling uses DDIM-based gradient updates
"""

import math
from typing import Dict, List, Optional, Tuple

import torch


class IaNDiffusion:
    """
    Image and Noise (IaN) diffusion implementation.

    This class implements a diffusion process that simultaneously estimates
    image and noise components using trigonometric interpolation.
    """

    def __init__(
        self,
        timestep_respacing: Optional[int] = None,
        loss_type: str = "l2",
        num_timesteps: int = 1000,
    ):
        if num_timesteps < 1:
            raise ValueError("num_timesteps must be positive")

        self.loss_type = loss_type
        self.num_timesteps = num_timesteps
        self.timestep_respacing = self._resolve_timestep_respacing(timestep_respacing)

    def _resolve_timestep_respacing(self, timestep_respacing: Optional[int]) -> int:
        if timestep_respacing is None:
            return self.num_timesteps

        respacing = int(timestep_respacing)
        if not 1 <= respacing <= self.num_timesteps:
            raise ValueError(
                "timestep_respacing must be between 1 and num_timesteps "
                f"(got {respacing} for {self.num_timesteps})"
            )
        return respacing

    @staticmethod
    def _broadcast_timesteps(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return t.to(device=x.device, dtype=x.dtype).view(x.shape[:1] + (1,) * (x.ndim - 1))

    def _angles(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._broadcast_timesteps(t, x) * (math.pi / 2 / self.num_timesteps)

    def _remaining_angles(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return (1 - self._broadcast_timesteps(t, x) / self.num_timesteps) * (math.pi / 2)

    def _sampling_sequence(self) -> List[int]:
        if self.timestep_respacing == 1:
            return [0]

        seq = torch.linspace(
            0,
            self.num_timesteps - 1,
            steps=self.timestep_respacing,
            dtype=torch.float64,
        )
        return seq.round().to(torch.long).unique_consecutive().tolist()

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion process: sample x_t from q(x_t | x_0).
        """
        if noise is None:
            noise = self.gen_noise(x_start)

        angles = self._angles(t, x_start)
        return torch.cos(angles) * x_start + torch.sin(angles) * noise

    def gen_noise(self, x_start: torch.Tensor, weight: float = 0.5) -> torch.Tensor:
        return torch.randn_like(x_start) * weight

    def training_losses(
        self,
        model: torch.nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        if model_kwargs is None:
            model_kwargs = {}

        noise = self.gen_noise(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        eps_recon, img_recon = model(x_t, t, **model_kwargs).chunk(2, dim=1)

        if self.loss_type != "l2":
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

        spatial_dims = tuple(range(1, x_start.ndim))
        return {
            "img_loss": (img_recon - x_start).pow(2).mean(dim=spatial_dims),
            "noise_loss": (eps_recon - noise).pow(2).mean(dim=spatial_dims),
        }

    def p_sample_loop(
        self,
        model: torch.nn.Module,
        shape: Tuple[int, ...],
        z: torch.Tensor,
        new_sampling: bool = False,
        model_kwargs: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if model_kwargs is None:
            model_kwargs = {}

        del shape  # Sampling follows the provided latent `z`.
        seq = self._sampling_sequence()

        if new_sampling:
            x0_preds, xs = self._predictor_corrector_steps(z, seq, model, model_kwargs)
        else:
            x0_preds, xs = self._generalized_steps(z, seq, model, model_kwargs)

        return xs, x0_preds[-1]

    @torch.no_grad()
    def _generalized_steps(
        self,
        x: torch.Tensor,
        seq: List[int],
        model: torch.nn.Module,
        model_kwargs: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if model_kwargs is None:
            model_kwargs = {}

        device = x.device
        batch_size = x.size(0)
        seq_next = [0] + seq[:-1]
        current = x.to(device)
        x0_preds, xs = [], [current.cpu()]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            next_t = torch.full((batch_size,), j, device=device, dtype=torch.long)

            angle = self._remaining_angles(t, current)
            next_angle = self._remaining_angles(next_t, current)
            eps_recon, img_recon = model(current, t, **model_kwargs).chunk(2, dim=1)

            current = current - (angle - next_angle) * (
                torch.cos(angle) * img_recon - torch.sin(angle) * eps_recon
            )

            x0_preds.append(img_recon.cpu())
            xs.append(current.cpu())

        return x0_preds, xs

    @torch.no_grad()
    def _predictor_corrector_steps(
        self,
        x: torch.Tensor,
        seq: List[int],
        model: torch.nn.Module,
        model_kwargs: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if model_kwargs is None:
            model_kwargs = {}

        predictor_stride = 2
        device = x.device
        batch_size = x.size(0)
        seq_next = [0] + seq[:-1]
        xt = x.to(device)
        xs, x0_preds = [xt.cpu()], []
        step_width = (math.pi / 2) / self.num_timesteps

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            next_t = torch.full((batch_size,), j, device=device, dtype=torch.long)
            h = (j - i) * step_width

            et, x0_t = model(xt, t, **model_kwargs).chunk(2, dim=1)
            beta_t = self._angles(t, xt)
            slope = torch.sin(beta_t) * x0_t - torch.cos(beta_t) * et

            predicted_step = i - predictor_stride * (i - j)
            if predicted_step > 0:
                xt_pred = xt - predictor_stride * h * slope
                t_pred = torch.full((batch_size,), predicted_step, device=device, dtype=torch.long)

                beta_corr = self._angles(next_t, xt_pred)
                beta_pred = self._angles(t_pred, xt_pred)
                alpha = torch.cos(beta_corr) / torch.cos(beta_pred)
                alpha_1 = torch.sqrt(torch.clamp(1 - alpha.square(), min=0.0))
                xt_next = alpha * xt_pred + alpha_1 * self.gen_noise(xt_pred)
            else:
                xt_next = xt - h * slope

            x0_preds.append(x0_t.cpu())
            xs.append(xt_next.cpu())

        return x0_preds, xs
