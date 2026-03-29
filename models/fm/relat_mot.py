from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.fm.DiT import DiTBlock, FinalLayer, TriplaneRoPE
from models.fm.utils import timestep_embedding


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


class MoTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        use_qknorm=True,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=True,
        fused_attn=True,
        use_rope=False,
        frames=8,
        input_size=32,
    ):
        super().__init__()
        self.rgb_branch = DiTBlock(
            hidden_size,
            num_heads,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rmsnorm=use_rmsnorm,
            wo_shift=wo_shift,
            fused_attn=fused_attn,
        )
        self.depth_branch = DiTBlock(
            hidden_size,
            num_heads,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rmsnorm=use_rmsnorm,
            wo_shift=wo_shift,
            fused_attn=fused_attn,
        )
        self.to_shared_rgb = nn.Linear(hidden_size, hidden_size)
        self.to_shared_depth = nn.Linear(hidden_size, hidden_size)
        self.shared_block = DiTBlock(
            hidden_size,
            num_heads,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rmsnorm=use_rmsnorm,
            wo_shift=wo_shift,
            fused_attn=fused_attn,
        )
        self.from_shared_rgb = nn.Linear(hidden_size, hidden_size)
        self.from_shared_depth = nn.Linear(hidden_size, hidden_size)
        self.feat_rope = TriplaneRoPE(hidden_size // num_heads, frames, input_size) if use_rope else None

    def forward(self, rgb, depth, t_rgb, t_depth, t_shared):
        rgb = self.rgb_branch(rgb, t_rgb, feat_rope=self.feat_rope)
        depth = self.depth_branch(depth, t_depth, feat_rope=self.feat_rope)

        shared_rgb = self.to_shared_rgb(rgb)
        shared_depth = self.to_shared_depth(depth)
        shared = torch.cat([shared_rgb, shared_depth], dim=1)
        shared = self.shared_block(shared, t_shared, feat_rope=None)
        shared_rgb, shared_depth = torch.split(shared, [rgb.size(1), depth.size(1)], dim=1)

        rgb = rgb + self.from_shared_rgb(shared_rgb)
        depth = depth + self.from_shared_depth(shared_depth)
        return rgb, depth


class ReLaTMoT(nn.Module):
    def __init__(
        self,
        input_size=32,
        in_channels=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        frames=8,
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
        self.use_checkpoint = use_checkpoint

        self.len_xy = input_size * input_size
        self.len_yt = frames * input_size
        self.len_xt = frames * input_size
        self.seq_len = self.len_xy + self.len_yt + self.len_xt
        self.ae_emb_dim = self.seq_len

        self.rgb_embedder = nn.Linear(in_channels * 2, hidden_size)
        self.depth_embedder = nn.Linear(in_channels * 2, hidden_size)
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, hidden_size))
        self.segment_embed = nn.Parameter(torch.zeros(1, 3, hidden_size))
        self.modality_embed = nn.Embedding(2, hidden_size)

        self.blocks = nn.ModuleList(
            [
                MoTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    use_qknorm=use_qknorm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                    wo_shift=wo_shift,
                    fused_attn=fused_attn,
                    use_rope=use_rope,
                    frames=frames,
                    input_size=input_size,
                )
                for _ in range(depth)
            ]
        )

        self.rgb_relation_head = TokenProjectionHead(hidden_size, rgb_teacher_dim)
        self.depth_relation_head = TokenProjectionHead(hidden_size, depth_teacher_dim)
        self.rgb_final = FinalLayer(hidden_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.depth_final = FinalLayer(hidden_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.segment_embed, std=0.02)
        nn.init.trunc_normal_(self.modality_embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        for block in self.blocks:
            nn.init.constant_(block.rgb_branch.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.rgb_branch.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.depth_branch.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.depth_branch.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.shared_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.shared_block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.rgb_final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.rgb_final.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.rgb_final.linear.weight, 0)
        nn.init.constant_(self.rgb_final.linear.bias, 0)
        nn.init.constant_(self.depth_final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.depth_final.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.depth_final.linear.weight, 0)
        nn.init.constant_(self.depth_final.linear.bias, 0)

    def _segment_indices(self, device):
        return torch.cat(
            [
                torch.zeros(self.len_xy, device=device).long(),
                torch.ones(self.len_yt, device=device).long(),
                torch.full((self.len_xt,), 2, device=device).long(),
            ]
        )

    def _prepare_tokens(self, x, cond, embedder, modality_index):
        if cond is None:
            cond = torch.zeros_like(x)
        x = torch.cat([x, cond], dim=1).transpose(1, 2)
        x = embedder(x)
        x = x + self.pos_embed[:, :x.size(1)]
        x = x + self.segment_embed[:, self._segment_indices(x.device)]
        modality = self.modality_embed.weight[modality_index].view(1, 1, -1)
        return x + modality

    def _time_embeddings(self, t_rgb, t_depth):
        t_rgb_emb = self.t_embedder(timestep_embedding(t_rgb, self.pos_embed.size(-1)).to(self.pos_embed.dtype))
        t_depth_emb = self.t_embedder(timestep_embedding(t_depth, self.pos_embed.size(-1)).to(self.pos_embed.dtype))
        t_shared = 0.5 * (t_rgb_emb + t_depth_emb)
        return t_rgb_emb, t_depth_emb, t_shared

    def _run_block(self, block, rgb, depth, t_rgb_emb, t_depth_emb, t_shared):
        if self.use_checkpoint and self.training:
            return checkpoint(
                lambda rgb_, depth_, t_rgb_, t_depth_, t_shared_: block(rgb_, depth_, t_rgb_, t_depth_, t_shared_),
                rgb,
                depth,
                t_rgb_emb,
                t_depth_emb,
                t_shared,
                use_reentrant=False,
            )
        return block(rgb, depth, t_rgb_emb, t_depth_emb, t_shared)

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
        hidden_rgb = None
        hidden_depth = None
        for index, block in enumerate(self.blocks, start=1):
            rgb, depth = self._run_block(block, rgb, depth, t_rgb_emb, t_depth_emb, t_shared)
            if index == self.aligned_depth:
                hidden_rgb = rgb
                hidden_depth = depth

        v_rgb = self.rgb_final(rgb, t_rgb_emb).transpose(1, 2)
        v_depth = self.depth_final(depth, t_depth_emb).transpose(1, 2)
        outputs = {"v_rgb": v_rgb, "v_depth": v_depth}
        if return_features and hidden_rgb is not None and hidden_depth is not None:
            outputs["hidden_rgb"] = hidden_rgb
            outputs["hidden_depth"] = hidden_depth
            outputs["aligned_rgb"] = self.rgb_relation_head(hidden_rgb)
            outputs["aligned_depth"] = self.depth_relation_head(hidden_depth)
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
