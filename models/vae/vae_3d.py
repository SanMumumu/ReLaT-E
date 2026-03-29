from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.std = torch.zeros_like(self.mean)
            self.var = torch.zeros_like(self.mean)

    def sample(self):
        if self.deterministic:
            return self.mean
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other=None):
        if self.deterministic:
            return torch.zeros(self.mean.size(0), device=self.mean.device, dtype=self.mean.dtype)
        if other is None:
            return 0.5 * torch.sum(
                self.mean.pow(2) + self.var - 1.0 - self.logvar,
                dim=(1, 2, 3, 4),
            )
        return 0.5 * torch.sum(
            (self.mean - other.mean).pow(2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
            dim=(1, 2, 3, 4),
        )

    def mode(self):
        return self.mean


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        return x + residual


class Conv3DAutoencoder(nn.Module):
    """
    Small dense 3D VAE with fixed temporal/spatial compression.
    Default compression for 8x128x128 clips is 4x16x16.
    """

    def __init__(self, embed_dim, vaeconfig):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.frames = int(vaeconfig["frames"])
        resolution = vaeconfig["resolution"]
        if isinstance(resolution, int):
            self.res_h = int(resolution)
            self.res_w = int(resolution)
        else:
            self.res_h = int(resolution[0])
            self.res_w = int(resolution[1])

        self.temporal_downsample = int(vaeconfig.get("latent_temporal_downsample", 2))
        self.spatial_downsample = int(vaeconfig.get("latent_spatial_downsample", 8))
        if self.frames % self.temporal_downsample != 0:
            raise ValueError("frames must be divisible by latent_temporal_downsample for Conv3DAutoencoder.")
        if self.res_h % self.spatial_downsample != 0 or self.res_w % self.spatial_downsample != 0:
            raise ValueError("resolution must be divisible by latent_spatial_downsample for Conv3DAutoencoder.")

        self.latent_frames = self.frames // self.temporal_downsample
        self.latent_h = self.res_h // self.spatial_downsample
        self.latent_w = self.res_w // self.spatial_downsample
        if self.latent_h != self.latent_w:
            raise ValueError("Current ReLaT-E generator assumes square latent spatial maps.")
        self.latent_size = self.latent_h
        self.ae_emb_dim = self.latent_frames * self.latent_h * self.latent_w

        in_channels = int(vaeconfig["in_channels"])
        out_channels = int(vaeconfig["out_channels"])
        base_channels = int(vaeconfig.get("channels", 128))
        c1 = max(base_channels // 4, 32)
        c2 = max(base_channels // 2, 64)
        c3 = base_channels

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=3, padding=1),
            ResidualBlock3D(c1, c1),
            nn.Conv3d(c1, c2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            ResidualBlock3D(c2, c2),
            nn.Conv3d(c2, c3, kernel_size=4, stride=2, padding=1),
            ResidualBlock3D(c3, c3),
            nn.Conv3d(c3, c3, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            ResidualBlock3D(c3, c3),
        )
        self.to_moments = nn.Conv3d(c3, 2 * self.embed_dim, kernel_size=1)

        self.from_latent = nn.Conv3d(self.embed_dim, c3, kernel_size=1)
        self.decoder = nn.Sequential(
            ResidualBlock3D(c3, c3),
            nn.ConvTranspose3d(c3, c3, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            ResidualBlock3D(c3, c3),
            nn.ConvTranspose3d(c3, c2, kernel_size=4, stride=2, padding=1),
            ResidualBlock3D(c2, c2),
            nn.ConvTranspose3d(c2, c1, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            ResidualBlock3D(c1, c1),
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv3d(c1, out_channels, kernel_size=3, padding=1),
        )
        self.kl_weight = float(vaeconfig.get("kl_weight", 1e-6))

    def encode(self, x, return_flat=False, sample_posterior=True):
        moments = self.to_moments(self.encoder(x))
        posterior = DiagonalGaussianDistribution(moments)
        z = posterior.sample() if sample_posterior else posterior.mode()
        kl_loss = posterior.kl().mean() * self.kl_weight
        if return_flat:
            return z.flatten(2), kl_loss
        return z, kl_loss

    def decode(self, z):
        if z.dim() == 3:
            z = z.view(z.size(0), z.size(1), self.latent_frames, self.latent_h, self.latent_w)
        z = self.from_latent(z)
        return torch.tanh(self.decoder(z))

    def forward(self, input, return_extract=False):
        z, kl_loss = self.encode(input, return_flat=False, sample_posterior=True)
        recon = self.decode(z)
        if return_extract:
            return recon, kl_loss, z.flatten(2)
        return recon, kl_loss

    def extract(self, x):
        moments = self.to_moments(self.encoder(x))
        z = DiagonalGaussianDistribution(moments).mode()
        return z.flatten(2)

    def decode_from_sample(self, h):
        return self.decode(h)
