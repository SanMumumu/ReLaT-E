from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelatEFlowMatching(nn.Module):
    def __init__(self, sampling_timesteps=50, sigma_min=1e-5, time_scale_factor=1000.0, same_noise=True):
        super().__init__()
        self.sampling_timesteps = int(sampling_timesteps)
        self.sigma_min = float(sigma_min)
        self.time_scale_factor = float(time_scale_factor)
        self.same_noise = bool(same_noise)

    def sample_training_tuple(
        self,
        z_rgb,
        z_depth,
        t_rgb=None,
        t_depth=None,
        noise_rgb=None,
        noise_depth=None,
        same_noise=None,
    ):
        same_noise = self.same_noise if same_noise is None else same_noise
        b = z_rgb.size(0)
        device = z_rgb.device
        if t_rgb is None:
            t_rgb = torch.rand((b,), device=device, dtype=z_rgb.dtype)
        if t_depth is None:
            t_depth = t_rgb if same_noise else torch.rand((b,), device=device, dtype=z_depth.dtype)
        if noise_rgb is None:
            noise_rgb = torch.randn_like(z_rgb)
        if noise_depth is None:
            noise_depth = noise_rgb if same_noise and noise_rgb.shape == z_depth.shape else torch.randn_like(z_depth)

        t_rgb_exp = t_rgb.view(b, 1, 1)
        t_depth_exp = t_depth.view(b, 1, 1)
        z_rgb_t = (1.0 - t_rgb_exp) * noise_rgb + t_rgb_exp * z_rgb
        z_depth_t = (1.0 - t_depth_exp) * noise_depth + t_depth_exp * z_depth
        target_rgb = z_rgb - noise_rgb
        target_depth = z_depth - noise_depth
        return {
            "t_rgb": t_rgb,
            "t_depth": t_depth,
            "noise_rgb": noise_rgb,
            "noise_depth": noise_depth,
            "z_rgb_t": z_rgb_t,
            "z_depth_t": z_depth_t,
            "target_rgb": target_rgb,
            "target_depth": target_depth,
            "scaled_t_rgb": t_rgb * self.time_scale_factor,
            "scaled_t_depth": t_depth * self.time_scale_factor,
        }

    def compute_loss(self, outputs, targets):
        loss_rgb = F.mse_loss(outputs["v_rgb"], targets["target_rgb"])
        loss_depth = F.mse_loss(outputs["v_depth"], targets["target_depth"])
        return loss_rgb + loss_depth, {
            "flow_loss_rgb": loss_rgb.detach(),
            "flow_loss_depth": loss_depth.detach(),
        }

    @torch.no_grad()
    def sample(self, generator, batch_size, latent_shape, cond_rgb=None, cond_depth=None, guidance_scale=1.0):
        device = next(generator.parameters()).device
        z_rgb = torch.randn((batch_size,) + latent_shape, device=device)
        if self.same_noise:
            z_depth = z_rgb.clone()
        else:
            z_depth = torch.randn((batch_size,) + latent_shape, device=device)

        dt = 1.0 / self.sampling_timesteps
        use_cfg = guidance_scale is not None and guidance_scale > 1.0 and cond_rgb is not None and cond_depth is not None
        if use_cfg:
            zero_rgb = torch.zeros_like(cond_rgb)
            zero_depth = torch.zeros_like(cond_depth)

        for index in range(self.sampling_timesteps):
            t_value = index / self.sampling_timesteps
            t_rgb = torch.full((batch_size,), t_value, device=device)
            t_depth = t_rgb
            scaled_t_rgb = t_rgb * self.time_scale_factor
            scaled_t_depth = t_depth * self.time_scale_factor

            if use_cfg:
                v_rgb_cond, v_depth_cond = generator.forward_sampling(
                    z_rgb, z_depth, cond_rgb, cond_depth, scaled_t_rgb, scaled_t_depth
                )
                v_rgb_uncond, v_depth_uncond = generator.forward_sampling(
                    z_rgb, z_depth, zero_rgb, zero_depth, scaled_t_rgb, scaled_t_depth
                )
                v_rgb = v_rgb_uncond + guidance_scale * (v_rgb_cond - v_rgb_uncond)
                v_depth = v_depth_uncond + guidance_scale * (v_depth_cond - v_depth_uncond)
            else:
                v_rgb, v_depth = generator.forward_sampling(
                    z_rgb, z_depth, cond_rgb, cond_depth, scaled_t_rgb, scaled_t_depth
                )

            z_rgb = z_rgb + v_rgb * dt
            z_depth = z_depth + v_depth * dt
        return z_rgb, z_depth
