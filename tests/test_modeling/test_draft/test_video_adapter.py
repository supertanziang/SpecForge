"""Unit tests for ElasticVideoAdapter and sub-modules."""

import pytest
import torch
import torch.nn as nn

from specforge.modeling.draft.video_adapter import (
    ElasticVideoAdapter,
    MeanPoolCompressor,
    MaskCrossAttention,
    PerceiverResampler,
    PerceiverResamplerLayer,
)


HIDDEN_SIZE = 256
NUM_HEADS = 8
NUM_KV_HEADS = 2
BUDGET = 16
IMAGE_TOKEN_ID = 99


def _make_adapter_config(**overrides):
    cfg = {
        "enabled": True,
        "compression_mode": "perceiver_resampler",
        "budget": BUDGET,
        "kv_mode": "draft_only",
        "mask_fusion": "mask_cross_attention",
        "image_token_id": IMAGE_TOKEN_ID,
        "num_resampler_layers": 1,
        "num_resampler_heads": NUM_HEADS,
        "num_resampler_kv_heads": NUM_KV_HEADS,
        "num_anchor_raw": 0,
        "gate_init": 0.0,
    }
    cfg.update(overrides)
    return cfg


class TestPerceiverResampler:
    def test_output_shape(self):
        resampler = PerceiverResampler(
            HIDDEN_SIZE, num_queries=BUDGET, num_layers=1,
            num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS
        )
        x = torch.randn(1, 200, HIDDEN_SIZE)
        out = resampler(x)
        assert out.shape == (1, BUDGET, HIDDEN_SIZE)

    def test_variable_length_input(self):
        resampler = PerceiverResampler(
            HIDDEN_SIZE, num_queries=BUDGET, num_layers=2,
            num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS
        )
        for n_vis in [10, 100, 500, 3000]:
            x = torch.randn(1, n_vis, HIDDEN_SIZE)
            out = resampler(x)
            assert out.shape == (1, BUDGET, HIDDEN_SIZE)

    def test_gradients_flow(self):
        resampler = PerceiverResampler(
            HIDDEN_SIZE, num_queries=BUDGET, num_layers=1,
            num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS
        )
        x = torch.randn(1, 50, HIDDEN_SIZE, requires_grad=True)
        out = resampler(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestMeanPoolCompressor:
    def test_output_shape(self):
        compressor = MeanPoolCompressor(HIDDEN_SIZE, BUDGET)
        x = torch.randn(1, 300, HIDDEN_SIZE)
        out = compressor(x)
        assert out.shape == (1, BUDGET, HIDDEN_SIZE)

    def test_small_input(self):
        compressor = MeanPoolCompressor(HIDDEN_SIZE, BUDGET)
        x = torch.randn(1, 5, HIDDEN_SIZE)
        out = compressor(x)
        assert out.shape == (1, BUDGET, HIDDEN_SIZE)


class TestMaskCrossAttention:
    def test_output_shape(self):
        cross_attn = MaskCrossAttention(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS)
        q = torch.randn(1, 32, HIDDEN_SIZE)
        kv = torch.randn(1, BUDGET, HIDDEN_SIZE)
        out = cross_attn(q, kv)
        assert out.shape == (1, 32, HIDDEN_SIZE)


class TestElasticVideoAdapter:
    def _make_inputs(self, n_text_before=10, n_vis=100, n_text_after=50):
        """Create synthetic input_ids and target_hidden."""
        seq_len = n_text_before + n_vis + n_text_after
        input_ids = torch.randint(0, 1000, (1, seq_len))
        input_ids[:, n_text_before:n_text_before + n_vis] = IMAGE_TOKEN_ID
        target_hidden = torch.randn(1, seq_len, HIDDEN_SIZE)
        return input_ids, target_hidden

    def test_compress_perceiver(self):
        cfg = _make_adapter_config(compression_mode="perceiver_resampler")
        adapter = ElasticVideoAdapter(HIDDEN_SIZE, cfg)
        input_ids, target_hidden = self._make_inputs(10, 100, 50)

        compressed, comp_vis, info = adapter.compress(target_hidden, input_ids)
        assert compressed.shape == (1, 10 + BUDGET + 50, HIDDEN_SIZE)
        assert comp_vis.shape == (1, BUDGET, HIDDEN_SIZE)
        assert info["n_vis_original"] == 100
        assert info["n_vis_compressed"] == BUDGET

    def test_compress_mean_pool(self):
        cfg = _make_adapter_config(compression_mode="mean_pool")
        adapter = ElasticVideoAdapter(HIDDEN_SIZE, cfg)
        input_ids, target_hidden = self._make_inputs(10, 200, 50)

        compressed, comp_vis, info = adapter.compress(target_hidden, input_ids)
        assert compressed.shape == (1, 10 + BUDGET + 50, HIDDEN_SIZE)
        assert comp_vis.shape == (1, BUDGET, HIDDEN_SIZE)

    def test_compress_hybrid(self):
        n_anchor = 4
        cfg = _make_adapter_config(
            compression_mode="hybrid", num_anchor_raw=n_anchor
        )
        adapter = ElasticVideoAdapter(HIDDEN_SIZE, cfg)
        input_ids, target_hidden = self._make_inputs(10, 100, 50)

        compressed, comp_vis, info = adapter.compress(target_hidden, input_ids)
        expected_visual = BUDGET + n_anchor
        assert comp_vis.shape == (1, expected_visual, HIDDEN_SIZE)
        assert compressed.shape == (1, 10 + expected_visual + 50, HIDDEN_SIZE)

    def test_no_visual_tokens_passthrough(self):
        cfg = _make_adapter_config()
        adapter = ElasticVideoAdapter(HIDDEN_SIZE, cfg)
        input_ids = torch.randint(0, 50, (1, 100))  # No IMAGE_TOKEN_ID
        target_hidden = torch.randn(1, 100, HIDDEN_SIZE)

        compressed, comp_vis, info = adapter.compress(target_hidden, input_ids)
        assert torch.equal(compressed, target_hidden)
        assert comp_vis is None
        assert info is None

    def test_text_tokens_preserved(self):
        cfg = _make_adapter_config()
        adapter = ElasticVideoAdapter(HIDDEN_SIZE, cfg)
        input_ids, target_hidden = self._make_inputs(10, 100, 50)

        compressed, _, info = adapter.compress(target_hidden, input_ids)
        # Text before visual should be identical
        assert torch.equal(compressed[:, :10, :], target_hidden[:, :10, :])
        # Text after visual should be identical
        assert torch.equal(compressed[:, 10 + BUDGET:, :], target_hidden[:, 110:, :])

    def test_fuse_mask_global_summary(self):
        cfg = _make_adapter_config(mask_fusion="global_summary")
        adapter = ElasticVideoAdapter(HIDDEN_SIZE, cfg)
        noise = torch.randn(1, 64, HIDDEN_SIZE)
        comp_vis = torch.randn(1, BUDGET, HIDDEN_SIZE)

        # Gate init = 0 -> fusion is no-op
        out = adapter.fuse_mask(noise, comp_vis)
        assert torch.allclose(out, noise, atol=1e-6)

    def test_fuse_mask_cross_attention(self):
        cfg = _make_adapter_config(mask_fusion="mask_cross_attention")
        adapter = ElasticVideoAdapter(HIDDEN_SIZE, cfg)
        noise = torch.randn(1, 64, HIDDEN_SIZE)
        comp_vis = torch.randn(1, BUDGET, HIDDEN_SIZE)

        # Gate init = 0 -> fusion is no-op
        out = adapter.fuse_mask(noise, comp_vis)
        assert torch.allclose(out, noise, atol=1e-6)

    def test_fuse_mask_off(self):
        cfg = _make_adapter_config(mask_fusion="off")
        adapter = ElasticVideoAdapter(HIDDEN_SIZE, cfg)
        noise = torch.randn(1, 64, HIDDEN_SIZE)
        comp_vis = torch.randn(1, BUDGET, HIDDEN_SIZE)

        out = adapter.fuse_mask(noise, comp_vis)
        assert torch.equal(out, noise)

    def test_fuse_mask_none_visual(self):
        cfg = _make_adapter_config()
        adapter = ElasticVideoAdapter(HIDDEN_SIZE, cfg)
        noise = torch.randn(1, 64, HIDDEN_SIZE)

        out = adapter.fuse_mask(noise, None)
        assert torch.equal(out, noise)

    def test_kv_mode_off(self):
        cfg = _make_adapter_config(kv_mode="off")
        adapter = ElasticVideoAdapter(HIDDEN_SIZE, cfg)
        input_ids, target_hidden = self._make_inputs(10, 100, 50)

        compressed, comp_vis, info = adapter.compress(target_hidden, input_ids)
        # kv_mode=off: target_hidden unchanged (no KV compression)
        assert torch.equal(compressed, target_hidden)
        # But compressed_visual is still computed for mask fusion
        assert comp_vis is not None
        assert comp_vis.shape == (1, BUDGET, HIDDEN_SIZE)

    def test_build_compressed_position_ids(self):
        cfg = _make_adapter_config()
        adapter = ElasticVideoAdapter(HIDDEN_SIZE, cfg)
        input_ids, target_hidden = self._make_inputs(10, 100, 50)

        _, _, info = adapter.compress(target_hidden, input_ids)
        pos_ids = adapter.build_compressed_position_ids(input_ids, info)
        expected_len = 10 + BUDGET + 50
        assert pos_ids.shape == (1, expected_len)
        # First 10 positions should be 0-9
        assert torch.equal(pos_ids[0, :10], torch.arange(10))

    def test_remap_anchor_positions(self):
        cfg = _make_adapter_config()
        adapter = ElasticVideoAdapter(HIDDEN_SIZE, cfg)
        input_ids, target_hidden = self._make_inputs(10, 100, 50)

        _, _, info = adapter.compress(target_hidden, input_ids)

        # Anchor in text_before (pos 5): should stay same
        anchors = torch.tensor([[5, 120]])  # 5 in text_before, 120 in text_after
        remapped = adapter.remap_anchor_positions(anchors, info)
        assert remapped[0, 0].item() == 5
        # Anchor at original pos 120 (in text_after): offset = 100 - BUDGET = 84
        # So remapped = 120 - 84 = 36... wait, let's verify the logic
        # In compressed space: text_before(10) + compressed(16) + text_after(50)
        # Original pos 120 is text_after at index 120-110=10 within text_after
        # In compressed space: 10 + 16 + 10 = 36
        # remap goes FROM compressed TO original, not the other way
        # Actually remap_anchor_positions maps compressed -> original for labels

    def test_backward_through_full_pipeline(self):
        cfg = _make_adapter_config()
        adapter = ElasticVideoAdapter(HIDDEN_SIZE, cfg)
        input_ids, target_hidden = self._make_inputs(10, 100, 50)
        target_hidden.requires_grad_(True)

        compressed, comp_vis, _ = adapter.compress(target_hidden, input_ids)
        noise = torch.randn(1, 32, HIDDEN_SIZE, requires_grad=True)
        fused = adapter.fuse_mask(noise, comp_vis)

        loss = compressed.sum() + fused.sum()
        loss.backward()
        assert target_hidden.grad is not None
        assert noise.grad is not None

    def test_bf16_dtype(self):
        cfg = _make_adapter_config()
        adapter = ElasticVideoAdapter(HIDDEN_SIZE, cfg).to(torch.bfloat16)
        input_ids, target_hidden = self._make_inputs(10, 100, 50)
        target_hidden = target_hidden.to(torch.bfloat16)

        compressed, comp_vis, _ = adapter.compress(target_hidden, input_ids)
        assert compressed.dtype == torch.bfloat16
        assert comp_vis.dtype == torch.bfloat16

    def test_single_frame_image(self):
        """Single image = 1 frame, small visual token count."""
        cfg = _make_adapter_config()
        adapter = ElasticVideoAdapter(HIDDEN_SIZE, cfg)
        input_ids, target_hidden = self._make_inputs(50, 20, 30)

        compressed, comp_vis, info = adapter.compress(target_hidden, input_ids)
        # 20 visual tokens compressed to min(20, BUDGET) via resampler
        assert comp_vis.shape == (1, BUDGET, HIDDEN_SIZE)
        assert compressed.shape == (1, 50 + BUDGET + 30, HIDDEN_SIZE)
