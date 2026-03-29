from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.fm.DiT import DiTBlock, FinalLayer
from models.fm.utils import timestep_embedding


class VolumeRoPE(nn.Module):
    def __init__(self, head_dim, frames, input_size, base=10000):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be divisible by 2"

        d_t = (head_dim // 3) // 2 * 2
        d_y = (head_dim // 3) // 2 * 2
        d_x = head_dim - d_t - d_y
        if d_x % 2 != 0:
            d_x -= 1
            d_y += 1
        if min(d_t, d_y, d_x) <= 0:
            raise ValueError("head_dim is too small for 3D rotary embedding.")

        inv_freq_t = 1.0 / (base ** (torch.arange(0, d_t, 2).float() / d_t))
        inv_freq_y = 1.0 / (base ** (torch.arange(0, d_y, 2).float() / d_y))
        inv_freq_x = 1.0 / (base ** (torch.arange(0, d_x, 2).float() / d_x))

        t_coords = torch.arange(frames).view(frames, 1, 1).expand(frames, input_size, input_size).reshape(-1).float()
        y_coords = torch.arange(input_size).view(1, input_size, 1).expand(frames, input_size, input_size).reshape(-1).float()
        x_coords = torch.arange(input_size).view(1, 1, input_size).expand(frames, input_size, input_size).reshape(-1).float()

        freqs_half = torch.cat(
            [
                torch.outer(t_coords, inv_freq_t),
                torch.outer(y_coords, inv_freq_y),
                torch.outer(x_coords, inv_freq_x),
            ],
            dim=-1,
        )
        freqs = torch.cat([freqs_half, freqs_half], dim=-1)
        self.register_buffer("cos_cached", freqs.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", freqs.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        cos = self.cos_cached[:, :, : x.size(2), :]
        sin = self.sin_cached[:, :, : x.size(2), :]
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)


class TokenProjectionHead(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_size, out_channels, kernel_size=1),
        )

    def forward(self, hidden):
        return self.net(hidden.transpose(1, 2)).transpose(1, 2)


class DiTStream(nn.Module):
    def __init__(
        self,
        hidden_size,
        depth,
        num_heads,
        use_qknorm=True,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=True,
        fused_attn=True,
        use_rope=False,
        frames=4,
        input_size=16,
        use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    use_qknorm=use_qknorm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                    wo_shift=wo_shift,
                    fused_attn=fused_attn,
                )
                for _ in range(depth)
            ]
        )
        self.feat_rope = VolumeRoPE(hidden_size // num_heads, frames, input_size) if use_rope else None

    def _run_block(self, block, x, t_emb):
        if self.use_checkpoint and self.training:
            return checkpoint(
                lambda x_, t_: block(x_, t_, feat_rope=self.feat_rope),
                x,
                t_emb,
                use_reentrant=False,
            )
        return block(x, t_emb, feat_rope=self.feat_rope)

    def forward(self, x, t_emb, capture_layer=None):
        hidden = None
        for index, block in enumerate(self.blocks, start=1):
            x = self._run_block(block, x, t_emb)
            if capture_layer is not None and index == capture_layer:
                hidden = x
        return x, hidden


class ReLaTMoT(nn.Module):
    def __init__(
        self,
        input_size=16,
        in_channels=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        frames=4,
        aligned_depth=4,
        rgb_teacher_dim=768,
        depth_teacher_dim=768,
        use_qknorm=True,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=True,
        fused_attn=True,
        use_rope=False,
        same_noise=True,
        use_checkpoint=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.frames = frames
        self.input_size = input_size
        self.aligned_depth = aligned_depth
        self.same_noise = same_noise

        self.seq_len = frames * input_size * input_size
        self.ae_emb_dim = self.seq_len

        self.rgb_embedder = nn.Linear(in_channels * 2, hidden_size)
        self.depth_embedder = nn.Linear(in_channels * 2, hidden_size)
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, hidden_size))
        self.modality_embed = nn.Embedding(2, hidden_size)

        self.rgb_stream = DiTStream(
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rmsnorm=use_rmsnorm,
            wo_shift=wo_shift,
            fused_attn=fused_attn,
            use_rope=use_rope,
            frames=frames,
            input_size=input_size,
            use_checkpoint=use_checkpoint,
        )
        self.depth_stream = DiTStream(
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rmsnorm=use_rmsnorm,
            wo_shift=wo_shift,
            fused_attn=fused_attn,
            use_rope=use_rope,
            frames=frames,
            input_size=input_size,
            use_checkpoint=use_checkpoint,
        )
        self.to_shared_rgb = nn.Linear(hidden_size, hidden_size)
        self.to_shared_depth = nn.Linear(hidden_size, hidden_size)
        self.shared_stream = DiTStream(
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rmsnorm=use_rmsnorm,
            wo_shift=wo_shift,
            fused_attn=fused_attn,
            use_rope=False,
            frames=frames,
            input_size=input_size,
            use_checkpoint=use_checkpoint,
        )
        self.from_shared_rgb = nn.Linear(hidden_size, hidden_size)
        self.from_shared_depth = nn.Linear(hidden_size, hidden_size)

        self.rgb_relation_head = TokenProjectionHead(hidden_size, rgb_teacher_dim)
        self.depth_relation_head = TokenProjectionHead(hidden_size, depth_teacher_dim)
        self.rgb_final = FinalLayer(hidden_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.depth_final = FinalLayer(hidden_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.modality_embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        for stream in (self.rgb_stream, self.depth_stream, self.shared_stream):
            for block in stream.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.rgb_final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.rgb_final.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.rgb_final.linear.weight, 0)
        nn.init.constant_(self.rgb_final.linear.bias, 0)
        nn.init.constant_(self.depth_final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.depth_final.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.depth_final.linear.weight, 0)
        nn.init.constant_(self.depth_final.linear.bias, 0)

    def _prepare_tokens(self, x, cond, embedder, modality_index):
        if cond is None:
            cond = torch.zeros_like(x)
        x = torch.cat([x, cond], dim=1).transpose(1, 2)
        x = embedder(x)
        x = x + self.pos_embed[:, : x.size(1)]
        modality = self.modality_embed.weight[modality_index].view(1, 1, -1)
        return x + modality

    def _time_embeddings(self, t_rgb, t_depth):
        t_rgb_emb = self.t_embedder(timestep_embedding(t_rgb, self.pos_embed.size(-1)).to(self.pos_embed.dtype))
        t_depth_emb = self.t_embedder(timestep_embedding(t_depth, self.pos_embed.size(-1)).to(self.pos_embed.dtype))
        t_shared = 0.5 * (t_rgb_emb + t_depth_emb)
        return t_rgb_emb, t_depth_emb, t_shared

    def _fuse_streams(self, rgb, depth, t_shared):
        shared = torch.cat([self.to_shared_rgb(rgb), self.to_shared_depth(depth)], dim=1)
        shared, _ = self.shared_stream(shared, t_shared, capture_layer=None)
        shared_rgb, shared_depth = torch.split(shared, [rgb.size(1), depth.size(1)], dim=1)
        rgb = rgb + self.from_shared_rgb(shared_rgb)
        depth = depth + self.from_shared_depth(shared_depth)
        return rgb, depth

    def forward(
        self,
        z_rgb_t,
        z_depth_t,
        cond_rgb=None,
        cond_depth=None,
        t_rgb=None,
        t_depth=None,
        return_features=True,
    ):
        if t_rgb is None:
            t_rgb = torch.zeros(z_rgb_t.size(0), device=z_rgb_t.device, dtype=z_rgb_t.dtype)
        if t_depth is None:
            t_depth = t_rgb
        t_rgb_emb, t_depth_emb, t_shared = self._time_embeddings(t_rgb, t_depth)

        rgb = self._prepare_tokens(z_rgb_t, cond_rgb, self.rgb_embedder, modality_index=0)
        depth = self._prepare_tokens(z_depth_t, cond_depth, self.depth_embedder, modality_index=1)

        rgb, hidden_rgb = self.rgb_stream(rgb, t_rgb_emb, capture_layer=self.aligned_depth)
        depth, hidden_depth = self.depth_stream(depth, t_depth_emb, capture_layer=self.aligned_depth)
        rgb, depth = self._fuse_streams(rgb, depth, t_shared)

        v_rgb = self.rgb_final(rgb, t_rgb_emb).transpose(1, 2)
        v_depth = self.depth_final(depth, t_depth_emb).transpose(1, 2)
        outputs = {"v_rgb": v_rgb, "v_depth": v_depth}
        if return_features:
            aligned_rgb = hidden_rgb if hidden_rgb is not None else rgb
            aligned_depth = hidden_depth if hidden_depth is not None else depth
            outputs["hidden_rgb"] = aligned_rgb
            outputs["hidden_depth"] = aligned_depth
            outputs["aligned_rgb"] = self.rgb_relation_head(aligned_rgb)
            outputs["aligned_depth"] = self.depth_relation_head(aligned_depth)
        return outputs

    def forward_sampling(self, z_rgb_t, z_depth_t, cond_rgb, cond_depth, t_rgb, t_depth):
        outputs = self.forward(
            z_rgb_t=z_rgb_t,
            z_depth_t=z_depth_t,
            cond_rgb=cond_rgb,
            cond_depth=cond_depth,
            t_rgb=t_rgb,
            t_depth=t_depth,
            return_features=False,
        )
        return outputs["v_rgb"], outputs["v_depth"]
