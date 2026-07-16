"""Diffusion model factory and loader for PRDiT.

Provides :func:`loading_diffusion` (the primary entry point) and
:func:`create_diffusion` (the lower-level Gaussian-diffusion constructor).

Adapted from OpenAI's diffusion repositories:

- GLIDE: https://github.com/openai/glide-text2im
- ADM:   https://github.com/openai/guided-diffusion
- IDDPM: https://github.com/openai/improved-diffusion
"""

from .flow_matching import FlowMatching
from .ian_diffusion import IaNDiffusion

# NOTE: ``gaussian_diffusion`` / ``SpacedDiffusion`` (the legacy Gaussian
# path) are imported lazily inside ``create_diffusion`` so that the default
# flow-matching path (out_channels==1) does not depend on them.


DEFAULT_TIMESTEP_RESPACING = "1000"


def loading_diffusion(config, rank: int = 0):
    """Instantiate the correct generative process from the experiment config.

    Selects :class:`~diffusion.flow_matching.FlowMatching` (single velocity
    head) when ``config.model.out_channels == 1`` — the default method for this
    project. ``out_channels == 2`` keeps the legacy dual-head
    :class:`~diffusion.ian_diffusion.IaNDiffusion` for reference.

    Parameters
    ----------
    config : Config
        Experiment configuration object.
    rank : int, optional
        Process rank; informational prints execute only on rank 0 (default ``0``).

    Returns
    -------
    FlowMatching or IaNDiffusion
        Configured generative-process instance.
    """
    out_channels = config.model.out_channels

    if out_channels == 1:
        num_steps = int(getattr(config.model, "num_sampling_steps", 100))
        if rank == 0:
            print(f"Loading the FlowMatching process (velocity head, {num_steps} sampling steps)")
        return FlowMatching(num_sampling_steps=num_steps, loss_type="l2")

    if out_channels == 2:
        if rank == 0:
            print("Loading the legacy IaNDiffusion model (dual image+noise head)")
        return IaNDiffusion(
            timestep_respacing=DEFAULT_TIMESTEP_RESPACING,
            loss_type="l2",
        )

    raise ValueError(f"Unsupported out_channels: {out_channels}")


def create_diffusion(
    timestep_respacing,
    noise_schedule: str = "linear",
    use_kl: bool = False,
    sigma_small: bool = False,
    predict_xstart: bool = False,
    learn_sigma: bool = True,
    rescale_learned_sigmas: bool = False,
    diffusion_steps: int = 1000,
) -> "SpacedDiffusion":
    """Construct a :class:`SpacedDiffusion` from named schedule parameters.

    Parameters
    ----------
    timestep_respacing : str or list
        Comma-separated step counts per schedule section, or ``"ddimN"`` for
        DDIM fixed striding.
    noise_schedule : str, optional
        Beta schedule name (default ``"linear"``).
    use_kl : bool, optional
        Use rescaled KL loss (default ``False``).
    sigma_small : bool, optional
        Use fixed-small variance (default ``False``).
    predict_xstart : bool, optional
        Model predicts ``x_0`` instead of ``ε`` (default ``False``).
    learn_sigma : bool, optional
        Model outputs learned variance (default ``True``).
    rescale_learned_sigmas : bool, optional
        Rescale learned sigma loss (default ``False``).
    diffusion_steps : int, optional
        Total diffusion steps before spacing (default ``1000``).

    Returns
    -------
    SpacedDiffusion
        Configured diffusion instance.
    """
    from . import gaussian_diffusion as gd
    from .respace import SpacedDiffusion, space_timesteps

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    if timestep_respacing in (None, ""):
        timestep_respacing = [diffusion_steps]
        
    # the space_timesteps return {0,1,...,998,999}
    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.START_X if predict_xstart else gd.ModelMeanType.EPSILON
        ),
        model_var_type=(
            gd.ModelVarType.LEARNED_RANGE
            if learn_sigma
            else (
                gd.ModelVarType.FIXED_SMALL
                if sigma_small
                else gd.ModelVarType.FIXED_LARGE
            )
        ),
        loss_type=loss_type,
    )
    
    return diffusion
