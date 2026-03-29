"""
Design the new diffusion operations, similar to gaussian diffusion function.
1. ian means image and noise. 
2. The model estimates the image and noise at the same time.
3. The xt = image * cos(t/T * pi/2) + noise * sin(t/T * pi/2)
4. Loss become |image - \hat{x_0}| + |noise - \hat{noise}|
5. The sampling process is based on DDIM sample, using the gradient update.
"""
import math

import numpy as np
import torch as th
import enum

from diffusion.gaussian_diffusion import mean_flat


class IaNDiffusion:
    def __init__(
        self,
        timestep_respacing=None,
        loss_type = "l2",
        num_timesteps=1000,
    ):
        self.loss_type = loss_type
        self.num_timesteps = num_timesteps
        self.timestep_respacing = int(timestep_respacing)
    
    def gen_noise(self, input_data, weight=0.5):
        if isinstance(input_data, (tuple, list, th.Size)):
            # input is (shape)
            return th.randn(*input_data) * weight
        else:
            # input is Tensor (x_start)
            return th.randn_like(input_data) * weight
    
    def q_sample(self, x_start, t):
        noise = self.gen_noise(x_start)

        t_reshape = t.reshape(x_start.shape[:1] + (1,) * (x_start.ndim - 1)).to(x_start.device)
        x_t = th.cos(t_reshape/self.num_timesteps * math.pi/2) * x_start + \
              th.sin(t_reshape/self.num_timesteps * math.pi/2) * noise
        return x_t
    
    def training_losses(self, model, x_start, t, 
                      model_kwargs=None, y=None):
        if model_kwargs is None:
            model_kwargs = {}
        
        # half the noise energy
        noise = self.gen_noise(x_start)
            
        t_reshape = t.reshape(x_start.shape[:1] + (1,) * (x_start.ndim - 1)).to(x_start.device)
        
        x_t = th.cos(t_reshape/self.num_timesteps * math.pi/2) * x_start + \
              th.sin(t_reshape/self.num_timesteps * math.pi/2) * noise
        
        terms = {}
        eps_recon, img_recon = model(x_t, t, **model_kwargs).chunk(2, dim=1)
        assert x_t.shape == x_start.shape
        assert eps_recon.shape == noise.shape
        assert eps_recon.shape == x_start.shape
        if self.loss_type == "l2":
            terms["img_loss"] = (img_recon - x_start).pow(2).mean(dim=list(range(1,len(x_start.shape))))
            terms["noise_loss"] = (eps_recon - noise).pow(2).mean(dim=list(range(1,len(x_start.shape))))
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

        # assert terms["loss"].shape == th.Size([B])
        return terms
    
    def p_sample_loop(self, model, shape, z, 
                      clip_denoised=False, progress=False, 
                      device=None, new_sampling=False, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        
        if z is not None:
            x = z
        else:
            x = self.gen_noise(shape)
        # x = x + th.randn_like(x) * 0.01
        
        skip = self.num_timesteps // self.timestep_respacing
        seq = range(0, self.num_timesteps, skip)
        
        if new_sampling:
            print("Using the new sampling approach....")
            x0_preds, xs = self.generalized_steps_new(x, seq, model, model_kwargs)
        else:
            print("Using the standard gradient update approach....")
            x0_preds, xs = self.generalized_steps(x, seq, model, model_kwargs)
        
        return xs, x0_preds[-1]

    @th.no_grad()
    def generalized_steps(self, x, seq, model, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        n = x.size(0)
        seq_next = [0] + list(seq[:-1])
        x0_preds, xs = [], [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (th.ones(n) * i).to(x.device)
            next_t = (th.ones(n) * j).to(x.device)
            at = (1 - (t / 1000)[:, None, None, None, None])
            next_at = (1 - (next_t / 1000)[:, None, None, None, None])
            
            xt = xs[-1].to(x.device)
            eps_recon, img_recon = model(xt, t, **model_kwargs).chunk(2, dim=1)
            img_recon = th.clamp(img_recon, -1, 1)
            xt_next = xt - (at - next_at) * math.pi/2 *(
                th.cos(at*math.pi/2) * img_recon - th.sin(at * math.pi/2) * eps_recon
            )
            
            x0_preds.append(img_recon.to('cpu'))
            xs.append(xt_next.to('cpu'))
        return x0_preds, xs
    
    @th.no_grad()
    def generalized_steps_new(self, x, seq, model, b, **kwargs):
        """
        x:       initial tensor [B, C, D, H, W]
        seq:     list of timesteps, e.g. [0, 10, 20, ..., 1000]
        model:   your diffusion net, returns (e_t, x0_t) when called as model(x_t, t)
        forward_step: how many discrete steps to jump backwards in the predictor (p)
        backward_step: how many discrete steps to jump forwards from predictor to land at t-1 (c)
        """
        p = 2
        min_f, max_f = 1, p
        n = x.size(0)
        seq_next = [0] + list(seq[:-1])
        xs, x0_preds = [x], []

        half_pi = th.pi / 2
        step_w  = th.pi / 2000

        for i, j in zip(reversed(seq), reversed(seq_next)):
            B = x.device
            t      = th.full((n,), i,     device=B)
            next_t = th.full((n,), j,     device=B)
            h      = (j - i) * step_w  # negative scalar
            step_size = (i-j)

            xt = xs[-1].to(B)
            et, x0_t = model(xt, t).chunk(2, dim=1)
            #x0_t = th.clamp(x0_t, -1, 1)
            # print(f"{i}: et: {et.shape}, {et.std(dim=(1,2,3,4), unbiased=False).mean()}")
            
            # slope f(t) = sin(θ_t)*x0_t - cos(θ_t)*e_t
            beta_t = (t / 1000)[..., None, None, None, None] * half_pi
            f_t = th.sin(beta_t) * x0_t - th.cos(beta_t) * et
            
            # 1) Predictor: jump back by p
            if i - p*(i-j) > 0:
                xt_pred = xt - p * h * f_t  # lands at t - p*(i-j)
                t_pred  = th.full((n,), i - p*(i-j), device=B)
                # 2) Corrector: exact noise injection from t_pred -> t_corr
                t_corr = th.full((n,), i - (i-j), device=B)
                beta_corr = (t_corr / 1000)[..., None, None, None, None] * half_pi
                beta_pred = (t_pred  / 1000)[..., None, None, None, None] * half_pi

                alpha = th.cos(beta_corr) / th.cos(beta_pred)
                alpha_1 = th.sqrt(th.clamp(1 - alpha**2, min=0.0))

                # final transition with fresh noise
                # noise_scale = th.clamp(et.std(dim=(1,2,3,4), unbiased=False), 0.0, 1.0).view(-1, 1, 1, 1, 1)
                xt_next = alpha * xt_pred + alpha_1 * self.gen_noise(xt_pred)
            else:
                xt_next = xt - h * f_t
            # record
            x0_preds.append(x0_t.cpu())
            xs.append(xt_next.cpu())

        return x0_preds, xs
