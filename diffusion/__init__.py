# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from .ian_diffusion import IaNDiffusion
from .respace import SpacedDiffusion, space_timesteps


DEFAULT_TIMESTEP_RESPACING = "1000"


def loading_diffusion(config, rank=0):
    out_channels = config.model.out_channels

    if out_channels == 1:
        if rank == 0:
            print("Loading the standard diffusion model")
        return create_diffusion(
            noise_schedule=config.data.schedule_type,
            learn_sigma=False,
            timestep_respacing=DEFAULT_TIMESTEP_RESPACING,
        )

    if out_channels == 2:
        if rank == 0:
            print("Loading the IaNDiffusion model")
        return IaNDiffusion(
            timestep_respacing=DEFAULT_TIMESTEP_RESPACING,
            loss_type="l2",
        )

    raise ValueError(f"Unsupported out_channels: {out_channels}")


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
):
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
