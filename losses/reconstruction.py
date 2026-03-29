from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.lpips import LPIPS


class AutoencoderReconstructionLoss(nn.Module):
    def __init__(self, mse_weight=1.0, l1_weight=0.0, perceptual_weight=0.0, kl_weight=1.0, channels=3):
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.l1_weight = float(l1_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.kl_weight = float(kl_weight)
        self.channels = int(channels)
        self.perceptual = None
        if self.perceptual_weight > 0 and self.channels == 3:
            self.perceptual = LPIPS().eval()
            for param in self.perceptual.parameters():
                param.requires_grad = False

    def forward(self, inputs, reconstructions, kl_loss):
        mse = F.mse_loss(reconstructions, inputs)
        l1 = F.l1_loss(reconstructions, inputs)

        if self.perceptual is not None:
            inputs_2d = inputs.transpose(1, 2).flatten(0, 1)
            recons_2d = reconstructions.transpose(1, 2).flatten(0, 1)
            perceptual = self.perceptual(recons_2d.contiguous(), inputs_2d.contiguous()).mean()
        else:
            perceptual = inputs.new_zeros(())

        total = (
            self.mse_weight * mse +
            self.l1_weight * l1 +
            self.perceptual_weight * perceptual +
            self.kl_weight * kl_loss
        )
        return total, {
            "mse": mse.detach(),
            "l1": l1.detach(),
            "perceptual": perceptual.detach(),
            "kl": kl_loss.detach(),
        }
