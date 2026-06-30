"""Elastic Video Vision Adapter for DFlash.

Compresses video visual tokens to a fixed budget and optionally fuses
compressed visual summary into the draft generation path.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm


class PerceiverResamplerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim)
        self.k_norm = Qwen3RMSNorm(self.head_dim)

        self.cross_attn_norm = Qwen3RMSNorm(hidden_size)
        self.kv_norm = Qwen3RMSNorm(hidden_size)
        self.post_attn_norm = Qwen3RMSNorm(hidden_size)
        self.mlp = Qwen3MLP(
            _DummyConfig(hidden_size=hidden_size, intermediate_size=hidden_size * 4)
        )

    def forward(
        self, queries: torch.Tensor, kv_input: torch.Tensor
    ) -> torch.Tensor:
        bsz, q_len, _ = queries.shape
        kv_len = kv_input.shape[1]

        residual = queries
        queries_normed = self.cross_attn_norm(queries)
        kv_normed = self.kv_norm(kv_input)

        q = self.q_proj(queries_normed).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(kv_normed).view(bsz, kv_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(kv_normed).view(bsz, kv_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_out = self.o_proj(attn_out)
        hidden = residual + attn_out

        residual = hidden
        hidden = self.post_attn_norm(hidden)
        hidden = self.mlp(hidden)
        hidden = residual + hidden
        return hidden


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_queries: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)
        self.layers = nn.ModuleList(
            [
                PerceiverResamplerLayer(hidden_size, num_heads, num_kv_heads)
                for _ in range(num_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(hidden_size)

    def forward(self, visual_hidden: torch.Tensor) -> torch.Tensor:
        bsz = visual_hidden.shape[0]
        queries = self.queries.expand(bsz, -1, -1)
        for layer in self.layers:
            queries = layer(queries, visual_hidden)
        return self.norm(queries)


class MeanPoolCompressor(nn.Module):
    def __init__(self, hidden_size: int, budget: int):
        super().__init__()
        self.budget = budget
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = Qwen3RMSNorm(hidden_size)

    def forward(self, visual_hidden: torch.Tensor) -> torch.Tensor:
        bsz, n_vis, h = visual_hidden.shape
        x = visual_hidden.transpose(1, 2)
        x = F.adaptive_avg_pool1d(x, self.budget)
        x = x.transpose(1, 2)
        return self.norm(self.proj(x))


class MaskCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim)
        self.k_norm = Qwen3RMSNorm(self.head_dim)

    def forward(
        self, query_hidden: torch.Tensor, kv_hidden: torch.Tensor
    ) -> torch.Tensor:
        bsz, q_len, _ = query_hidden.shape
        kv_len = kv_hidden.shape[1]

        q = self.q_proj(query_hidden).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(kv_hidden).view(bsz, kv_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(kv_hidden).view(bsz, kv_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(attn_out)


class ElasticVideoAdapter(nn.Module):
    def __init__(self, hidden_size: int, va_config: dict):
        super().__init__()
        self.hidden_size = hidden_size
        self.budget = va_config["budget"]
        self.compression_mode = va_config.get("compression_mode", "perceiver_resampler")
        self.kv_mode = va_config.get("kv_mode", "draft_only")
        self.mask_fusion_type = va_config.get("mask_fusion", "off")
        self.image_token_id = va_config["image_token_id"]
        self.num_anchor_raw = va_config.get("num_anchor_raw", 0)

        num_heads = va_config.get("num_resampler_heads", 28)
        num_kv_heads = va_config.get("num_resampler_kv_heads", 4)
        num_layers = va_config.get("num_resampler_layers", 2)

        if self.compression_mode in ("perceiver_resampler", "hybrid"):
            self.resampler = PerceiverResampler(
                hidden_size, self.budget, num_layers, num_heads, num_kv_heads
            )
        if self.compression_mode in ("mean_pool",):
            self.pool_compressor = MeanPoolCompressor(hidden_size, self.budget)

        if self.mask_fusion_type == "global_summary":
            self.summary_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            gate_init = va_config.get("gate_init", 0.0)
            # FSDP requires 1D tensors (no scalar params); use shape [1].
            self.fusion_gate = nn.Parameter(torch.tensor([gate_init]))
        elif self.mask_fusion_type == "mask_cross_attention":
            self.fusion_cross_attn = MaskCrossAttention(
                hidden_size, num_heads, num_kv_heads
            )
            gate_init = va_config.get("gate_init", 0.0)
            self.fusion_gate = nn.Parameter(torch.tensor([gate_init]))

    def compress(
        self,
        target_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """Compress visual tokens in target_hidden.

        Returns:
            compressed_target_hidden: [batch, S_compressed, H]
            compressed_visual: [batch, B(+K), H] or None
            compression_info: dict with remapping metadata, or None
        """
        visual_mask = input_ids == self.image_token_id  # [batch, S]
        has_visual = visual_mask.any()

        if not has_visual:
            return target_hidden, None, None

        bsz = target_hidden.shape[0]
        assert bsz == 1, "Video adapter only supports batch_size=1"

        mask_1d = visual_mask[0]  # [S]
        visual_tokens = target_hidden[:, mask_1d, :]  # [1, Nv, H]
        text_tokens = target_hidden[:, ~mask_1d, :]  # [1, Nt, H]

        n_vis = visual_tokens.shape[1]
        if n_vis == 0:
            return target_hidden, None, None

        if self.compression_mode == "perceiver_resampler":
            compressed_visual = self.resampler(visual_tokens)
        elif self.compression_mode == "mean_pool":
            compressed_visual = self.pool_compressor(visual_tokens)
        elif self.compression_mode == "hybrid":
            resampled = self.resampler(visual_tokens)
            if self.num_anchor_raw > 0:
                scores = visual_tokens.norm(dim=-1)  # [1, Nv]
                _, topk_idx = scores.topk(
                    min(self.num_anchor_raw, n_vis), dim=-1
                )
                topk_idx = topk_idx.sort(dim=-1).values
                anchor_tokens = torch.gather(
                    visual_tokens,
                    1,
                    topk_idx.unsqueeze(-1).expand(-1, -1, self.hidden_size),
                )
                compressed_visual = torch.cat([resampled, anchor_tokens], dim=1)
            else:
                compressed_visual = resampled
        else:
            return target_hidden, None, None

        # Reassemble: text_before + compressed_visual + text_after
        # Find the boundary: first and last visual token positions
        vis_indices = mask_1d.nonzero(as_tuple=True)[0]
        vis_start = vis_indices[0].item()
        vis_end = vis_indices[-1].item() + 1

        text_before = target_hidden[:, :vis_start, :]  # [1, vis_start, H]
        text_after = target_hidden[:, vis_end:, :]  # [1, S - vis_end, H]

        # Handle non-contiguous visual tokens (multiple image segments)
        # For simplicity: if visual tokens are non-contiguous, we still extract
        # them all, compress, and place the compressed block at the position of
        # the first visual segment. Text tokens between segments are kept in order.
        if _is_contiguous(vis_indices):
            compressed_target = torch.cat(
                [text_before, compressed_visual, text_after], dim=1
            )
        else:
            # Non-contiguous: rebuild preserving text between visual segments
            compressed_target = _rebuild_noncontiguous(
                target_hidden, mask_1d, compressed_visual
            )

        # Build position mapping info
        S_orig = target_hidden.shape[1]
        S_compressed = compressed_target.shape[1]
        compression_info = {
            "visual_mask": visual_mask,
            "vis_start": vis_start,
            "vis_end": vis_end,
            "n_vis_original": n_vis,
            "n_vis_compressed": compressed_visual.shape[1],
            "S_original": S_orig,
            "S_compressed": S_compressed,
        }

        if self.kv_mode == "off":
            return target_hidden, compressed_visual, compression_info

        return compressed_target, compressed_visual, compression_info

    def fuse_mask(
        self,
        noise_embedding: torch.Tensor,
        compressed_visual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if compressed_visual is None or self.mask_fusion_type == "off":
            return noise_embedding

        if self.mask_fusion_type == "global_summary":
            g_v = compressed_visual.mean(dim=1, keepdim=True)  # [1, 1, H]
            g_v = self.summary_proj(g_v)
            return noise_embedding + self.fusion_gate * g_v

        if self.mask_fusion_type == "mask_cross_attention":
            fusion_out = self.fusion_cross_attn(noise_embedding, compressed_visual)
            return noise_embedding + self.fusion_gate * fusion_out

        return noise_embedding

    def build_compressed_position_ids(
        self,
        input_ids: torch.Tensor,
        compression_info: dict,
    ) -> torch.Tensor:
        """Build position IDs for compressed context."""
        S_orig = compression_info["S_original"]
        vis_start = compression_info["vis_start"]
        vis_end = compression_info["vis_end"]
        n_compressed = compression_info["n_vis_compressed"]
        device = input_ids.device

        visual_mask = compression_info["visual_mask"][0]  # [S]

        if _is_contiguous(visual_mask.nonzero(as_tuple=True)[0]):
            pos_before = torch.arange(vis_start, device=device)
            pos_visual = torch.linspace(
                vis_start, vis_end - 1, n_compressed, device=device
            ).long()
            pos_after = torch.arange(vis_end, S_orig, device=device)
            return torch.cat([pos_before, pos_visual, pos_after]).unsqueeze(0)

        # Non-contiguous: text positions keep original, compressed visual gets
        # evenly spaced positions within original visual span
        text_positions = torch.arange(S_orig, device=device)[~visual_mask]
        vis_positions = torch.linspace(
            vis_start, vis_end - 1, n_compressed, device=device
        ).long()
        # Merge maintaining order
        all_pos = torch.cat([text_positions, vis_positions])
        all_pos = all_pos.sort().values
        return all_pos.unsqueeze(0)

    def remap_anchor_positions(
        self,
        anchor_positions: torch.Tensor,
        compression_info: dict,
    ) -> torch.Tensor:
        """Remap anchor positions from original to compressed coordinate space.

        Anchors are always at text positions (loss_mask=0 at visual positions).
        """
        visual_mask = compression_info["visual_mask"][0]  # [S]
        n_vis_original = compression_info["n_vis_original"]
        n_vis_compressed = compression_info["n_vis_compressed"]
        removed = n_vis_original - n_vis_compressed

        if removed == 0:
            return anchor_positions

        # For each anchor position, count how many visual tokens precede it
        # and subtract the difference (removed count up to that position)
        cum_vis = visual_mask.long().cumsum(0)  # [S]
        device = anchor_positions.device

        # anchors: [bsz, N_anchors]
        # Each anchor is at a text position. Its new position in compressed space:
        # new_pos = orig_pos - min(cum_vis[orig_pos], removed)
        # Because all removed tokens come from the visual span
        safe_anchors = anchor_positions.clamp(0, len(cum_vis) - 1)
        vis_before = cum_vis[safe_anchors]  # [bsz, N]
        # Visual tokens removed before this anchor = min(vis_before, n_vis_original) - budget portion
        # More precisely: all Nv tokens become n_vis_compressed tokens.
        # A text token at position p has cum_vis[p] visual tokens before it.
        # In compressed space, those cum_vis[p] visual tokens become
        # min(cum_vis[p], n_vis_compressed) compressed tokens (if we're past the
        # visual region, they've all been compressed).
        # offset_removed = cum_vis[p] - min(cum_vis[p], n_vis_compressed)
        offset_removed = (vis_before - vis_before.clamp(max=n_vis_compressed)).clamp(
            min=0
        )
        # But actually simpler: cum_vis at any text position after the visual
        # span = n_vis_original. Tokens before the span have cum_vis=0.
        # So offset = min(cum_vis[p], n_vis_original) - min(cum_vis[p], n_vis_compressed)
        # = cum_vis[p] - min(cum_vis[p], n_vis_compressed) when cum_vis[p] <= n_vis_original
        new_anchors = anchor_positions - offset_removed
        return new_anchors


def _is_contiguous(indices: torch.Tensor) -> bool:
    if len(indices) <= 1:
        return True
    return (indices[-1] - indices[0] + 1) == len(indices)


def _rebuild_noncontiguous(
    target_hidden: torch.Tensor,
    visual_mask_1d: torch.Tensor,
    compressed_visual: torch.Tensor,
) -> torch.Tensor:
    """Rebuild context with non-contiguous visual segments replaced by compressed tokens.

    Strategy: place all compressed tokens at the position of the first visual segment,
    keep all text tokens in their relative order.
    """
    S = target_hidden.shape[1]
    vis_indices = visual_mask_1d.nonzero(as_tuple=True)[0]
    first_vis = vis_indices[0].item()

    text_before = target_hidden[:, :first_vis, :]
    # Collect text tokens that appear after first visual token
    text_after_mask = ~visual_mask_1d.clone()
    text_after_mask[:first_vis] = False
    text_after = target_hidden[:, text_after_mask, :]

    return torch.cat([text_before, compressed_visual, text_after], dim=1)


class _DummyConfig:
    """Minimal config for Qwen3MLP initialization."""

    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = "silu"
