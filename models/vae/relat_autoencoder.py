from __future__ import annotations

import torch
import torch.nn as nn

from models.vae.vae_3d import Conv3DAutoencoder


class Relat3DVAE(nn.Module):
    def __init__(self, embed_dim, vaeconfig, bn_momentum=0.1):
        super().__init__()
        architecture = str(getattr(vaeconfig, "architecture", vaeconfig.get("architecture", "conv3d_vae")))
        if architecture not in {"conv3d_vae", "vit_3d_vae"}:
            raise ValueError(f"Unsupported VAE architecture for ReLaT-E: {architecture}")
        self.model = Conv3DAutoencoder(embed_dim, vaeconfig)
        self.frames = int(vaeconfig["frames"])
        self.latent_frames = int(self.model.latent_frames)
        self.latent_size = int(self.model.latent_size)
        self.ae_emb_dim = int(self.model.ae_emb_dim)
        self.latent_bn = nn.BatchNorm1d(embed_dim, eps=1e-4, momentum=bn_momentum, affine=False, track_running_stats=True)
        self.cond_latent_bn = nn.BatchNorm1d(embed_dim, eps=1e-4, momentum=bn_momentum, affine=False, track_running_stats=True)
        self.latent_bn.reset_running_stats()
        self.cond_latent_bn.reset_running_stats()
        self.bn_eps = 1e-4

    def _pad_frames(self, x):
        if x.size(2) == self.frames:
            return x
        if x.size(2) > self.frames:
            raise ValueError(f"Input has {x.size(2)} frames but autoencoder supports at most {self.frames}.")
        pad = self.frames - x.size(2)
        if x.size(2) == 0:
            raise ValueError("Cannot encode an empty sequence.")
        pad_frame = x[:, :, -1:].expand(-1, -1, pad, -1, -1)
        return torch.cat([x, pad_frame], dim=2)

    def _crop_frames(self, x, frames):
        if x.dim() == 4:
            batch = x.size(0) // self.frames
            x = x.view(batch, self.frames, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4).contiguous()
        return x[:, :, :frames]

    def _apply_bn(self, z, bn):
        mean = bn.running_mean.detach().clone().view(1, -1, 1).to(device=z.device, dtype=z.dtype)
        var = bn.running_var.detach().clone().view(1, -1, 1).to(device=z.device, dtype=z.dtype)
        return (z - mean) / torch.sqrt(var + self.bn_eps)

    def _apply_cond_bn(self, z):
        is_uncond = z.abs().amax(dim=(1, 2), keepdim=True) < 1e-12
        valid_mask = ~is_uncond.view(-1)
        if valid_mask.any():
            if valid_mask.all():
                z = self._apply_bn(z, self.cond_latent_bn)
            else:
                out = z.clone()
                valid_z = z[valid_mask]
                out[valid_mask] = self._apply_bn(valid_z, self.cond_latent_bn)
                z = out
        return z.masked_fill(is_uncond, 0.0)

    def encode_future(self, x, normalize=True):
        z = self.extract_latent(x)
        return self.normalize_latent(z) if normalize else z

    def encode_past(self, x, normalize=True):
        z = self.extract_latent(x)
        return self.normalize_latent(z, conditioning=True) if normalize else z

    def extract_latent(self, x):
        x = self._pad_frames(x)
        return self.model.extract(x)

    def normalize_latent(self, z, conditioning=False):
        if conditioning:
            return self._apply_cond_bn(z)
        return self._apply_bn(z, self.latent_bn)

    def denormalize_latent(self, z, conditioning=False):
        bn = self.cond_latent_bn if conditioning else self.latent_bn
        running_mean = bn.running_mean.view(1, -1, 1).to(device=z.device, dtype=z.dtype)
        running_var = bn.running_var.view(1, -1, 1).to(device=z.device, dtype=z.dtype)
        return z * torch.sqrt(running_var + self.bn_eps) + running_mean

    def forward_reconstruction(self, x):
        num_frames = x.size(2)
        x = self._pad_frames(x)
        recon, kl_loss, latent = self.model(x, return_extract=True)
        recon = self._crop_frames(recon, num_frames)
        return recon, kl_loss, latent

    def decode(self, z, num_frames=None, conditioning=False):
        z = self.denormalize_latent(z, conditioning=conditioning)
        recon = self.model.decode_from_sample(z)
        recon = self._crop_frames(recon, self.frames if num_frames is None else num_frames)
        return recon

    def forward(self, x=None, z=None, mode="reconstruct", normalize=False, conditioning=False, num_frames=None):
        if mode == "reconstruct":
            if x is None:
                raise ValueError("Input tensor x is required for reconstruct mode.")
            return self.forward_reconstruction(x)
        if mode == "extract":
            if x is None:
                raise ValueError("Input tensor x is required for extract mode.")
            latent = self.extract_latent(x)
            return self.normalize_latent(latent, conditioning=conditioning) if normalize else latent
        if mode == "decode":
            if z is None:
                raise ValueError("Latent tensor z is required for decode mode.")
            return self.decode(z, num_frames=num_frames, conditioning=conditioning)
        raise ValueError(f"Unsupported forward mode: {mode}")


RelatAutoencoder = Relat3DVAE
