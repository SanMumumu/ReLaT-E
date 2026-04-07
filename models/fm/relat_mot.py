from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.fm.DiT import DiTBlock, FinalLayer, RMSNorm, SwiGLUFFN, modulate
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


def _resolve_depth_hidden_size(hidden_size, num_heads, depth_width_ratio):
    if isinstance(depth_width_ratio, float) and not depth_width_ratio.is_integer():
        raise ValueError("depth_width_ratio must be an integer value.")
    depth_width_ratio = int(depth_width_ratio)
    if depth_width_ratio < 1:
        raise ValueError("depth_width_ratio must be >= 1.")
    if hidden_size % depth_width_ratio != 0:
        raise ValueError(f"hidden_size={hidden_size} must be divisible by depth_width_ratio={depth_width_ratio}.")
    depth_hidden_size = hidden_size // depth_width_ratio
    if depth_hidden_size % num_heads != 0:
        raise ValueError(
            f"depth_hidden_size={depth_hidden_size} must be divisible by num_heads={num_heads}; "
            "choose a depth_width_ratio that preserves head divisibility."
        )
    return depth_width_ratio, depth_hidden_size


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


class SharedMoTAttention(nn.Module):
    def __init__(
        self,
        in_dim,
        attn_dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        use_rmsnorm=False,
        fused_attn=True,
    ):
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.attn_dim = attn_dim
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(in_dim, attn_dim * 3, bias=qkv_bias)
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(attn_dim, in_dim)


class SharedMoTBranch(nn.Module):
    def __init__(
        self,
        hidden_size,
        attn_hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False,
        use_rmsnorm=False,
        wo_shift=False,
        fused_attn=True,
    ):
        super().__init__()
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        self.attn = SharedMoTAttention(
            hidden_size,
            attn_hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            fused_attn=fused_attn,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, hidden_size),
            )

        self.wo_shift = wo_shift
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )


class SharedMoTBlock(nn.Module):
    def __init__(
        self,
        rgb_hidden_size,
        depth_hidden_size,
        attn_hidden_size,
        num_heads,
        use_qknorm=True,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=True,
        fused_attn=True,
    ):
        super().__init__()
        self.rgb_block = SharedMoTBranch(
            rgb_hidden_size,
            attn_hidden_size,
            num_heads,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rmsnorm=use_rmsnorm,
            wo_shift=wo_shift,
            fused_attn=fused_attn,
        )
        self.depth_block = SharedMoTBranch(
            depth_hidden_size,
            attn_hidden_size,
            num_heads,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rmsnorm=use_rmsnorm,
            wo_shift=wo_shift,
            fused_attn=fused_attn,
        )

    @staticmethod
    def _modulation(block, t_emb):
        if block.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = block.adaLN_modulation(t_emb).chunk(4, dim=1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN_modulation(t_emb).chunk(6, dim=1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    @staticmethod
    def _project_qkv(block, x, feat_rope=None):
        attn = block.attn
        b, n_tokens, _ = x.shape
        qkv = attn.qkv(x).reshape(b, n_tokens, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)
        if feat_rope is not None:
            q = feat_rope(q)
            k = feat_rope(k)
        return q, k, v, attn.attn_dim

    @staticmethod
    def _shared_attention(attn, q, k, v):
        if attn.fused_attn and hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        q = q * attn.scale
        scores = q @ k.transpose(-2, -1)
        weights = scores.softmax(dim=-1)
        return weights @ v

    @staticmethod
    def _merge_heads(x, n_tokens, channels):
        return x.transpose(1, 2).reshape(x.size(0), n_tokens, channels)

    def forward(self, rgb, depth, t_rgb_emb, t_depth_emb, feat_rope=None):
        rgb_shift_msa, rgb_scale_msa, rgb_gate_msa, rgb_shift_mlp, rgb_scale_mlp, rgb_gate_mlp = self._modulation(
            self.rgb_block, t_rgb_emb
        )
        depth_shift_msa, depth_scale_msa, depth_gate_msa, depth_shift_mlp, depth_scale_mlp, depth_gate_mlp = self._modulation(
            self.depth_block, t_depth_emb
        )

        rgb_attn_in = modulate(self.rgb_block.norm1(rgb), rgb_shift_msa, rgb_scale_msa)
        depth_attn_in = modulate(self.depth_block.norm1(depth), depth_shift_msa, depth_scale_msa)
        q_rgb, k_rgb, v_rgb, channels = self._project_qkv(self.rgb_block, rgb_attn_in, feat_rope=feat_rope)
        q_depth, k_depth, v_depth, _ = self._project_qkv(self.depth_block, depth_attn_in, feat_rope=feat_rope)

        q = torch.cat([q_rgb, q_depth], dim=2)
        k = torch.cat([k_rgb, k_depth], dim=2)
        v = torch.cat([v_rgb, v_depth], dim=2)
        y = self._shared_attention(self.rgb_block.attn, q, k, v)
        y_rgb, y_depth = torch.split(y, [rgb.size(1), depth.size(1)], dim=2)

        y_rgb = self._merge_heads(y_rgb, rgb.size(1), channels)
        y_depth = self._merge_heads(y_depth, depth.size(1), channels)
        rgb = rgb + rgb_gate_msa.unsqueeze(1) * self.rgb_block.attn.proj(y_rgb)
        depth = depth + depth_gate_msa.unsqueeze(1) * self.depth_block.attn.proj(y_depth)

        rgb_mlp_in = modulate(self.rgb_block.norm2(rgb), rgb_shift_mlp, rgb_scale_mlp)
        depth_mlp_in = modulate(self.depth_block.norm2(depth), depth_shift_mlp, depth_scale_mlp)
        rgb = rgb + rgb_gate_mlp.unsqueeze(1) * self.rgb_block.mlp(rgb_mlp_in)
        depth = depth + depth_gate_mlp.unsqueeze(1) * self.depth_block.mlp(depth_mlp_in)
        return rgb, depth


class SharedMoTStream(nn.Module):
    def __init__(
        self,
        rgb_hidden_size,
        depth_hidden_size,
        attn_hidden_size,
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
                SharedMoTBlock(
                    rgb_hidden_size,
                    depth_hidden_size,
                    attn_hidden_size,
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
        self.feat_rope = VolumeRoPE(attn_hidden_size // num_heads, frames, input_size) if use_rope else None

    def _run_block(self, block, rgb, depth, t_rgb_emb, t_depth_emb):
        if self.use_checkpoint and self.training:
            return checkpoint(
                lambda rgb_, depth_, t_rgb_, t_depth_: block(
                    rgb_,
                    depth_,
                    t_rgb_,
                    t_depth_,
                    feat_rope=self.feat_rope,
                ),
                rgb,
                depth,
                t_rgb_emb,
                t_depth_emb,
                use_reentrant=False,
            )
        return block(rgb, depth, t_rgb_emb, t_depth_emb, feat_rope=self.feat_rope)

    def forward(self, rgb, depth, t_rgb_emb, t_depth_emb):
        for block in self.blocks:
            rgb, depth = self._run_block(block, rgb, depth, t_rgb_emb, t_depth_emb)
        return rgb, depth


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
        depth_width_ratio=1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.frames = frames
        self.input_size = input_size
        self.aligned_depth = aligned_depth
        self.same_noise = same_noise
        self.hidden_size = hidden_size
        self.depth_width_ratio, self.depth_hidden_size = _resolve_depth_hidden_size(
            hidden_size,
            num_heads,
            depth_width_ratio,
        )

        self.seq_len = frames * input_size * input_size
        self.ae_emb_dim = self.seq_len

        self.rgb_embedder = nn.Linear(in_channels * 2, hidden_size)
        self.depth_embedder = nn.Linear(in_channels * 2, self.depth_hidden_size)
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.depth_t_embedder = nn.Sequential(
            nn.Linear(self.depth_hidden_size, self.depth_hidden_size),
            nn.SiLU(),
            nn.Linear(self.depth_hidden_size, self.depth_hidden_size),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, hidden_size))
        self.depth_pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.depth_hidden_size))
        self.modality_embed = nn.Embedding(2, hidden_size)
        self.depth_modality_embed = nn.Parameter(torch.zeros(1, 1, self.depth_hidden_size))

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
            hidden_size=self.depth_hidden_size,
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
        self.to_shared_depth = nn.Linear(self.depth_hidden_size, self.depth_hidden_size)
        self.shared_stream = SharedMoTStream(
            rgb_hidden_size=hidden_size,
            depth_hidden_size=self.depth_hidden_size,
            attn_hidden_size=hidden_size,
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
        self.from_shared_depth = nn.Linear(self.depth_hidden_size, self.depth_hidden_size)

        self.rgb_relation_head = TokenProjectionHead(hidden_size, rgb_teacher_dim)
        self.depth_relation_head = TokenProjectionHead(self.depth_hidden_size, depth_teacher_dim)
        self.rgb_final = FinalLayer(hidden_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.depth_final = FinalLayer(self.depth_hidden_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.depth_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.modality_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.depth_modality_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        for stream in (self.rgb_stream, self.depth_stream):
            for block in stream.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.shared_stream.blocks:
            for branch in (block.rgb_block, block.depth_block):
                nn.init.constant_(branch.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(branch.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.rgb_final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.rgb_final.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.rgb_final.linear.weight, 0)
        nn.init.constant_(self.rgb_final.linear.bias, 0)
        nn.init.constant_(self.depth_final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.depth_final.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.depth_final.linear.weight, 0)
        nn.init.constant_(self.depth_final.linear.bias, 0)

    def _prepare_tokens(self, x, cond, embedder, pos_embed, modality_embed):
        if cond is None:
            cond = torch.zeros_like(x)
        x = torch.cat([x, cond], dim=1).transpose(1, 2)
        x = embedder(x)
        x = x + pos_embed[:, : x.size(1)]
        return x + modality_embed.view(1, 1, -1)

    def _time_embeddings(self, t_rgb, t_depth):
        t_rgb_emb = self.t_embedder(timestep_embedding(t_rgb, self.hidden_size).to(self.pos_embed.dtype))
        t_depth_emb = self.depth_t_embedder(timestep_embedding(t_depth, self.depth_hidden_size).to(self.depth_pos_embed.dtype))
        return t_rgb_emb, t_depth_emb

    def _fuse_streams(self, rgb, depth, t_rgb_emb, t_depth_emb):
        shared_rgb = self.to_shared_rgb(rgb)
        shared_depth = self.to_shared_depth(depth)
        shared_rgb, shared_depth = self.shared_stream(shared_rgb, shared_depth, t_rgb_emb, t_depth_emb)
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
        t_rgb_emb, t_depth_emb = self._time_embeddings(t_rgb, t_depth)

        rgb = self._prepare_tokens(
            z_rgb_t,
            cond_rgb,
            self.rgb_embedder,
            self.pos_embed,
            self.modality_embed.weight[0],
        )
        depth = self._prepare_tokens(
            z_depth_t,
            cond_depth,
            self.depth_embedder,
            self.depth_pos_embed,
            self.depth_modality_embed,
        )

        rgb, hidden_rgb = self.rgb_stream(rgb, t_rgb_emb, capture_layer=self.aligned_depth)
        depth, hidden_depth = self.depth_stream(depth, t_depth_emb, capture_layer=self.aligned_depth)
        rgb, depth = self._fuse_streams(rgb, depth, t_rgb_emb, t_depth_emb)

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
