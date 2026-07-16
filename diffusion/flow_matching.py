"""Flow Matching (rectified-flow / straight-line) process for PRDiT.

Unified definition used across this codebase
-------------------------------------------
Let ``x_noise ~ N(0, I)`` and ``x_data`` be a real sample, with time
``t ~ U(0, 1)``::

    x_t              = (1 - t) * x_noise + t * x_data
    target_velocity  = x_data - x_noise

Convention (deliberately avoids ``x0`` / ``x1`` naming):

- ``t = 0``  -> pure noise        (``x_t = x_noise``)
- ``t = 1``  -> real data         (``x_t = x_data``)
- the model predicts the **velocity** ``v_theta(x_t, t)`` directly,
  trained with ``L = || v_theta(x_t, t) - (x_data - x_noise) ||^2``.

Notes
-----
* Straight-line path with ``sigma_min = 0`` (the simplest baseline).
* ``t`` is continuous in ``[0, 1]``. Only when feeding the model's timestep
  embedder is ``t`` rescaled to the embedder's expected range
  (``model_t = t * embed_t_scale``, default ``1000``); the interpolation and
  velocity targets always use the raw ``t in [0, 1]``.
* There is **no** DDPM ``beta`` / ``alpha`` / ``alphas_cumprod`` / posterior
  logic here. ``num_timesteps`` of DDPM is unrelated to either the number of
  training iterations or the number of sampling steps.
* Sampling solves the ODE ``dx/dt = v_theta(x_t, t)`` from ``t = 0`` (noise)
  to ``t = 1`` (data). Euler is the default; Heun (2nd-order, 2 NFE/step) and
  RK4 (classic 4th-order, 4 NFE/step) are optional higher-order solvers.
  ``pc`` is a stochastic predictor-corrector (1 NFE/step, controlled by
  ``eta``) that re-randomises the noise component each step -- the flow-matching
  analogue of the DDPM/Restart renoising correctors. ``pc_heun`` (2 NFE/step)
  uses a Heun corrector as its deterministic backbone and only injects noise
  where the Euler-vs-Heun local error is large: the strength is
  ``eta_base * min(err/err_target, 1) * (1 - t)`` with ``err`` dt^2-normalised so
  ``err_target`` is step-count independent (``eta_base = 0`` recovers plain
  Heun). Report both ``num_steps`` and NFE (number of function evaluations).
"""

import math
from typing import Dict, List, Optional, Tuple

import torch


class FlowMatching:
    """Straight-line Flow Matching process with a single velocity head.

    Parameters
    ----------
    num_sampling_steps : int, optional
        Default number of ODE integration steps used by :meth:`sample`
        (default ``100``).
    loss_type : str, optional
        Velocity reconstruction loss; only ``"l2"`` is supported (default
        ``"l2"``).
    embed_t_scale : float, optional
        Scale applied to ``t in [0, 1]`` before it is passed to the model's
        timestep embedder (default ``1000.0``). This does **not** affect the
        interpolation or the velocity target.
    """

    def __init__(
        self,
        num_sampling_steps: int = 100,
        loss_type: str = "l2",
        embed_t_scale: float = 1000.0,
    ):
        if loss_type != "l2":
            raise NotImplementedError(f"Loss type '{loss_type}' is not implemented.")
        self.num_sampling_steps = int(num_sampling_steps)
        self.loss_type = loss_type
        self.embed_t_scale = float(embed_t_scale)

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _expand_t(t: torch.Tensor, ndim: int) -> torch.Tensor:
        """Reshape a ``[B]`` time tensor to broadcast over a volume tensor."""
        return t.reshape((-1,) + (1,) * (ndim - 1))

    def _model_t(self, t: torch.Tensor) -> torch.Tensor:
        """Map ``t in [0, 1]`` to the model's timestep-embedder input range."""
        return t * self.embed_t_scale

    def interpolate(
        self,
        x_data: torch.Tensor,
        x_noise: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the straight-line interpolant and its velocity target.

        Returns ``(x_t, target_velocity)`` with
        ``x_t = (1 - t) * x_noise + t * x_data`` and
        ``target_velocity = x_data - x_noise``.
        """
        t_b = self._expand_t(t, x_data.ndim).to(x_data.dtype)
        x_t = (1.0 - t_b) * x_noise + t_b * x_data
        target_velocity = x_data - x_noise
        return x_t, target_velocity

    # -- Training -------------------------------------------------------------

    def training_losses(
        self,
        model: torch.nn.Module,
        x_data: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the per-sample velocity-matching loss.

        Parameters
        ----------
        model : torch.nn.Module
            Velocity model returning a tensor with the same channel count as
            ``x_data``.
        x_data : torch.Tensor
            Clean input volumes of shape ``[B, C, D, H, W]``.
        t : torch.Tensor or None, optional
            Continuous times in ``[0, 1]`` of shape ``[B]``. Sampled uniformly
            when ``None`` (the standard Flow Matching choice).
        model_kwargs : dict or None, optional
            Extra keyword arguments forwarded to the model.

        Returns
        -------
        dict
            ``{"loss": Tensor[B], "velocity_loss": Tensor[B]}``
        """
        if model_kwargs is None:
            model_kwargs = {}

        batch_size = x_data.shape[0]
        if t is None:
            t = torch.rand(batch_size, device=x_data.device, dtype=x_data.dtype)
        else:
            t = t.to(device=x_data.device, dtype=x_data.dtype)

        x_noise = torch.randn_like(x_data)
        x_t, target_velocity = self.interpolate(x_data, x_noise, t)

        v_pred = model(x_t, self._model_t(t), **model_kwargs)

        spatial_dims = list(range(1, x_data.ndim))
        velocity_loss = (v_pred - target_velocity).pow(2).mean(dim=spatial_dims)
        return {"loss": velocity_loss, "velocity_loss": velocity_loss}

    # -- Reconstruction (used by depth-0 / coarse evaluation) -----------------

    @torch.no_grad()
    def predict_x_data(
        self,
        model: torch.nn.Module,
        x_data: torch.Tensor,
        t_value: float = 0.5,
        model_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate ``x_data`` from a noised sample via the predicted velocity.

        Given ``x_t`` at a single time ``t``, the straight-line model implies
        ``x_data = x_t + (1 - t) * v_theta(x_t, t)``. Returns
        ``(x_data_hat, x_t)``. Used as a cheap reconstruction metric for the
        coarse (depth-0) stage.
        """
        if model_kwargs is None:
            model_kwargs = {}

        batch_size = x_data.shape[0]
        t = torch.full((batch_size,), float(t_value), device=x_data.device, dtype=x_data.dtype)
        x_noise = torch.randn_like(x_data)
        x_t, _ = self.interpolate(x_data, x_noise, t)

        v_pred = model(x_t, self._model_t(t), **model_kwargs)
        x_data_hat = x_t + (1.0 - t_value) * v_pred
        return x_data_hat, x_t

    # -- Sampling -------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: Optional[Tuple[int, ...]] = None,
        x_noise: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        solver: str = "euler",
        eta: float = 0.0,
        err_target: float = 1.0,
        device: Optional[torch.device] = None,
        model_kwargs: Optional[Dict] = None,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        if model_kwargs is None:
            model_kwargs = {}
        if num_steps is None:
            num_steps = self.num_sampling_steps
        solver = solver.lower()
        if solver not in {"euler", "heun", "rk4", "pc", "pc_heun"}:
            raise ValueError(
                f"Unknown solver '{solver}'. Use 'euler', 'heun', 'rk4', 'pc', or 'pc_heun'."
            )
        eta = float(eta)
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"`eta` must be in [0, 1], got {eta}.")
        err_target = float(err_target)
        if err_target <= 0.0:
            raise ValueError(f"`err_target` must be > 0, got {err_target}.")

        if x_noise is None:
            if shape is None:
                raise ValueError("Either `shape` or `x_noise` must be provided.")
            x_noise = torch.randn(shape, device=device)

        x = x_noise
        device = x.device
        batch_size = x.shape[0]
        dt = 1.0 / num_steps

        trajectory: List[torch.Tensor] = [x.detach().cpu()] if return_trajectory else [x_noise]
        nfe = 0

        for step in range(num_steps):
            t_scalar = step * dt
            t = torch.full((batch_size,), t_scalar, device=device, dtype=x.dtype)
            v = model(x, self._model_t(t), **model_kwargs)
            nfe += 1

            if solver == "euler":
                x = x + dt * v
            elif solver == "heun":  # predictor + endpoint-velocity corrector
                x_pred = x + dt * v
                t_next = torch.full((batch_size,), min((step + 1) * dt, 1.0), device=device, dtype=x.dtype)
                v_next = model(x_pred, self._model_t(t_next), **model_kwargs)
                nfe += 1
                x = x + dt * 0.5 * (v + v_next)
            elif solver == "rk4":  # classic 4th-order Runge-Kutta (4 NFE/step)
                t_mid = torch.full((batch_size,), min(t_scalar + 0.5 * dt, 1.0), device=device, dtype=x.dtype)
                t_next = torch.full((batch_size,), min((step + 1) * dt, 1.0), device=device, dtype=x.dtype)
                k1 = v
                k2 = model(x + 0.5 * dt * k1, self._model_t(t_mid), **model_kwargs)
                k3 = model(x + 0.5 * dt * k2, self._model_t(t_mid), **model_kwargs)
                k4 = model(x + dt * k3, self._model_t(t_next), **model_kwargs)
                nfe += 3
                x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            elif solver == "pc":  # stochastic predictor-corrector (1 NFE/step)
                # Decompose the straight-line state exactly from the velocity:
                #   x = t * x_data + (1 - t) * x_noise
                # x_data is the Euler ("predictor") target; x_noise is the unit-
                # variance latent that the corrector partially re-randomises.
                x_data = x + (1.0 - t_scalar) * v
                x_noise = x - t_scalar * v
                if eta > 0.0:
                    # Variance-preserving mix keeps x_noise ~ N(0, I): this is the
                    # flow-matching analogue of DDPM/Restart noise re-injection.
                    fresh = torch.randn_like(x)
                    x_noise = math.sqrt(max(1.0 - eta * eta, 0.0)) * x_noise + eta * fresh
                t_next_scalar = min((step + 1) * dt, 1.0)
                # Reconstruct on the marginal at t_next. With eta = 0 this is
                # algebraically identical to the Euler update x + dt * v; the
                # final step (t_next = 1) lands exactly on x_data.
                x = t_next_scalar * x_data + (1.0 - t_next_scalar) * x_noise
            else:  # pc_heun: Heun corrector + err-gated adaptive stochastic refresh (2 NFE/step)
                t_next_scalar = min((step + 1) * dt, 1.0)
                t_next = torch.full((batch_size,), t_next_scalar, device=device, dtype=x.dtype)
                # -- Predictor (Euler) + deterministic Heun corrector --
                x_euler = x + dt * v
                v_next = model(x_euler, self._model_t(t_next), **model_kwargs)
                nfe += 1
                x_pc = x + 0.5 * dt * (v + v_next)

                if eta <= 0.0:
                    # eta_base = 0 -> exact Heun (no stochastic refresh).
                    x = x_pc
                else:
                    # -- Local error estimate (embedded Euler vs Heun) --
                    # x_pc - x_euler = 0.5*dt*(v_next - v) estimates Euler's O(dt^2)
                    # local truncation error. Per-sample RMS, relative, then
                    # dt^2-normalised so err_target is step-count independent.
                    dims = tuple(range(1, x.ndim))

                    def _rms(z):
                        return z.pow(2).mean(dim=dims, keepdim=True).sqrt()

                    err = _rms(x_pc - x_euler) / (_rms(x_pc) + 1e-6)
                    err_norm = err / (dt * dt)  # O(1), independent of step count

                    # -- Adaptive, curvature-gated noise strength --
                    # More noise where the field is more curved / error is larger;
                    # (1 - t_next) forces the refresh to vanish toward the data end.
                    gate = (err_norm / err_target).clamp(max=1.0)
                    eta_t = eta * gate * (1.0 - t_next_scalar)  # [B,1,...,1]

                    # -- Stochastic corrector on the Heun state --
                    # NOTE: v_next is measured at x_euler, not x_pc; using it to
                    # decompose x_pc is an O(dt^2) approximation that only affects
                    # the (small) eta>0 branch. Reconstruction is exact for eta_t=0.
                    x_data = x_pc + (1.0 - t_next_scalar) * v_next
                    x_noise = x_pc - t_next_scalar * v_next
                    fresh = torch.randn_like(x)
                    x_noise = torch.sqrt((1.0 - eta_t * eta_t).clamp(min=0.0)) * x_noise + eta_t * fresh
                    x = t_next_scalar * x_data + (1.0 - t_next_scalar) * x_noise

            if return_trajectory:
                trajectory.append(x.detach().cpu())

        return x, trajectory, nfe

    # -- Backward-compatible wrapper for the training / sampling scripts ------

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: torch.nn.Module,
        shape: Tuple[int, ...],
        z: torch.Tensor,
        clip_denoised: bool = False,
        progress: bool = False,
        new_sampling: bool = False,
        solver: Optional[str] = None,
        eta: float = 0.0,
        err_target: float = 1.0,
        device: Optional[torch.device] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Compatibility shim mirroring the old diffusion sampler interface.

        ``z`` is the initial ``N(0, I)`` noise at ``t = 0``. Pass ``solver``
        (``"euler"`` / ``"heun"`` / ``"rk4"`` / ``"pc"`` / ``"pc_heun"``) to pick
        the integrator; when it is ``None`` the legacy ``new_sampling`` flag
        selects Heun vs Euler. ``eta`` is the stochastic-corrector strength
        (``eta_base`` for ``pc_heun``) and ``err_target`` the dt^2-normalised
        error gate for ``pc_heun``. Returns ``(trajectory, final_sample)`` so
        that existing callers (``xs[-1]`` / final sample) keep working.
        """
        if solver is None:
            solver = "heun" if new_sampling else "euler"
        solver = solver.lower()
        nfe_per_step = {"euler": 1, "heun": 2, "rk4": 4, "pc": 1, "pc_heun": 2}.get(solver, 1)
        print(
            f"Flow-matching sampling: solver={solver}, "
            f"steps={self.num_sampling_steps}, "
            f"eta={eta}, err_target={err_target}, "
            f"NFE={nfe_per_step * self.num_sampling_steps}"
        )
        x_sample, trajectory, nfe = self.sample(
            model,
            x_noise=z,
            num_steps=self.num_sampling_steps,
            solver=solver,
            eta=eta,
            err_target=err_target,
            model_kwargs=model_kwargs,
            return_trajectory=True,
        )
        print(f"Flow-matching sampling done: NFE={nfe}")
        return trajectory, x_sample
