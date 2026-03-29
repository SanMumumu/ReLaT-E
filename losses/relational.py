from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalAlignmentLoss(nn.Module):
    def __init__(self, input_size, frames, beta_spatial=1.0, beta_temporal=1.0):
        super().__init__()
        self.input_size = int(input_size)
        self.frames = int(frames)
        self.beta_spatial = float(beta_spatial)
        self.beta_temporal = float(beta_temporal)
        self.len_xy = self.input_size * self.input_size
        self.len_yt = self.frames * self.input_size
        self.len_xt = self.frames * self.input_size

    def _infer_teacher_grid(self, num_teacher_tokens):
        candidate_ts = []
        for value in (self.frames, max(1, self.frames // 2), 1):
            if value not in candidate_ts:
                candidate_ts.append(value)

        for t_vfm in candidate_ts:
            if num_teacher_tokens % t_vfm != 0:
                continue
            spatial_tokens = num_teacher_tokens // t_vfm
            spatial_size = int(round(spatial_tokens ** 0.5))
            if spatial_size * spatial_size == spatial_tokens:
                return t_vfm, spatial_size, spatial_size

        best_triplet = None
        best_score = None
        max_t = min(num_teacher_tokens, max(self.frames, 1))
        for t_vfm in range(1, max_t + 1):
            if num_teacher_tokens % t_vfm != 0:
                continue
            spatial_tokens = num_teacher_tokens // t_vfm
            h_vfm = int(spatial_tokens ** 0.5)
            while h_vfm > 0:
                if spatial_tokens % h_vfm == 0:
                    w_vfm = spatial_tokens // h_vfm
                    score = abs(h_vfm - w_vfm) + abs(t_vfm - self.frames)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_triplet = (t_vfm, h_vfm, w_vfm)
                    break
                h_vfm -= 1
        if best_triplet is None:
            raise ValueError(f"Unable to infer a teacher grid for {num_teacher_tokens} tokens.")
        return best_triplet

    def _sample_triplane(self, student_tokens, teacher_tokens):
        b, _, teacher_dim = student_tokens.shape
        xy_plane = student_tokens[:, :self.len_xy, :].transpose(1, 2).reshape(b, teacher_dim, self.input_size, self.input_size)
        yt_plane = student_tokens[:, self.len_xy:self.len_xy + self.len_yt, :].transpose(1, 2).reshape(
            b, teacher_dim, self.frames, self.input_size
        )
        xt_plane = student_tokens[:, self.len_xy + self.len_yt:, :].transpose(1, 2).reshape(
            b, teacher_dim, self.frames, self.input_size
        )

        _, num_teacher_tokens, _ = teacher_tokens.shape
        t_vfm, h_vfm, w_vfm = self._infer_teacher_grid(num_teacher_tokens)
        t = torch.linspace(-1, 1, steps=t_vfm, device=student_tokens.device)
        y = torch.linspace(-1, 1, steps=h_vfm, device=student_tokens.device)
        x = torch.linspace(-1, 1, steps=w_vfm, device=student_tokens.device)
        grid_t, grid_y, grid_x = torch.meshgrid(t, y, x, indexing="ij")

        grid_xy = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2).expand(b, -1, -1, -1)
        grid_yt = torch.stack([grid_y, grid_t], dim=-1).view(1, 1, -1, 2).expand(b, -1, -1, -1)
        grid_xt = torch.stack([grid_x, grid_t], dim=-1).view(1, 1, -1, 2).expand(b, -1, -1, -1)

        feat_xy = F.grid_sample(xy_plane, grid_xy, mode="bilinear", align_corners=True).squeeze(2)
        feat_yt = F.grid_sample(yt_plane, grid_yt, mode="bilinear", align_corners=True).squeeze(2)
        feat_xt = F.grid_sample(xt_plane, grid_xt, mode="bilinear", align_corners=True).squeeze(2)
        student = (feat_xy + feat_yt + feat_xt).transpose(1, 2)
        return student, (t_vfm, h_vfm, w_vfm)

    def _standardize_teacher(self, teacher_tokens, grid):
        t_vfm, h_vfm, w_vfm = grid
        teacher = teacher_tokens.reshape(teacher_tokens.size(0), t_vfm, h_vfm * w_vfm, teacher_tokens.size(-1))
        spatial_mean = teacher.mean(dim=2, keepdim=True)
        spatial_std = teacher.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (teacher - spatial_mean) / spatial_std

    def _normalize_student(self, student_tokens, grid):
        t_vfm, h_vfm, w_vfm = grid
        return student_tokens.reshape(student_tokens.size(0), t_vfm, h_vfm * w_vfm, student_tokens.size(-1))

    def _spatial_loss(self, student, teacher):
        student = F.normalize(student, dim=-1)
        teacher = F.normalize(teacher, dim=-1)
        student_gram = torch.einsum("btid,btjd->btij", student, student)
        teacher_gram = torch.einsum("btid,btjd->btij", teacher, teacher)
        return (student_gram - teacher_gram).abs().mean()

    def _temporal_loss(self, student, teacher):
        student = F.normalize(student, dim=-1)
        teacher = F.normalize(teacher, dim=-1)
        losses = []
        for idx in range(student.size(1)):
            for jdx in range(student.size(1)):
                if idx == jdx:
                    continue
                student_rel = torch.einsum("bid,bjd->bij", student[:, idx], student[:, jdx])
                teacher_rel = torch.einsum("bid,bjd->bij", teacher[:, idx], teacher[:, jdx])
                losses.append((student_rel - teacher_rel).abs().mean())
        if not losses:
            return student.new_zeros(())
        return torch.stack(losses).mean()

    def forward(self, student_tokens, teacher_tokens):
        teacher_tokens = teacher_tokens.detach()
        aligned_student, grid = self._sample_triplane(student_tokens, teacher_tokens)
        teacher = self._standardize_teacher(teacher_tokens, grid)
        student = self._normalize_student(aligned_student, grid)

        spatial = self._spatial_loss(student, teacher)
        temporal = self._temporal_loss(student, teacher)
        total = self.beta_spatial * spatial + self.beta_temporal * temporal
        return total, {
            "spatial": spatial.detach(),
            "temporal": temporal.detach(),
        }
