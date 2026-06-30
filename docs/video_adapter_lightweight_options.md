# Video Adapter 轻量化方案对比

## 当前架构

```
compression_mode: perceiver_resampler (2层)
budget: 128
mask_fusion: global_summary
```

**数据流**：
```
vision tokens [1, 1760, H]
    ↓ PerceiverResampler (2层 cross-attn + 2层 MLP)
compressed [1, 128, H]
    ↓ 拼回 context → draft KV 从 2064 缩到 432
    ↓ 同时: mean(compressed) → summary_proj → gate × 加到 noise_embedding
```

**当前问题**：
- 压缩比 1.8x，接受率掉 6%，但 speedup 只有 0.94x（反而略慢）
- 原因：resampler 2层 forward 开销 > KV 缩短省的 attention 时间（单 batch 下）

---

## 可调参数一览

| 参数 | 当前值 | 作用 | 调小的代价 |
|---|---|---|---|
| `compression_mode` | perceiver_resampler | 压缩方式（复杂度最高） | 换 mean_pool 更快但压缩质量略差 |
| `num_resampler_layers` | 2 | resampler 层数 | 砍到 1 省一半计算 |
| `budget` | 128 | 压缩后 token 数 | 越小 KV 越短但信息损失越大 |
| `mask_fusion` | global_summary | 视觉摘要注入方式 | 关掉(off)省 Linear 计算 |
| `num_resampler_heads` | 28 | attention 头数 | 减少省计算，但表达力下降 |
| `num_resampler_kv_heads` | 4 | GQA kv 头数 | 已经很少，动空间不大 |
| `num_anchor_raw` | 0 | hybrid 模式保留的原始 token 数 | 0=纯压缩 |

---

## 三种轻量化方案

### 方案 A：极致轻量（推荐先试）

```json
{
  "compression_mode": "mean_pool",
  "budget": 32,
  "mask_fusion": "off"
}
```

**架构**：
```
vision tokens [1, 1760, H]
    ↓ adaptive_avg_pool1d(1760 → 32)
    ↓ Linear(H→H) + RMSNorm
compressed [1, 32, H]
    ↓ 拼回 → KV 从 2064 缩到 336
    ↓ fusion: 无
```

| 指标 | 预期 |
|---|---|
| adapter 计算 | 几乎为 0（一个 pooling + 一个 Linear） |
| 压缩比 | 1760→32 = **55x** |
| KV 缩短 | 2064 → 336（6.1x 短） |
| 接受率损失 | 可能 15-25%（压缩太狠信息损失大） |
| 参数量 | ~25M（1 个 Linear） |

**适用**：验证「极端压缩下接受率掉到多少」的下界。如果掉太多再往上调。

---

### 方案 B：折中（平衡速度与质量）

```json
{
  "compression_mode": "perceiver_resampler",
  "num_resampler_layers": 1,
  "budget": 64,
  "mask_fusion": "off"
}
```

**架构**：
```
vision tokens [1, 1760, H]
    ↓ PerceiverResampler (1层 cross-attn + MLP)
compressed [1, 64, H]
    ↓ 拼回 → KV 从 2064 缩到 368
    ↓ fusion: 无
```

| 指标 | 预期 |
|---|---|
| adapter 计算 | 当前的 1/4（1层 vs 2层 × budget 64 vs 128） |
| 压缩比 | 1760→64 = **27.5x** |
| KV 缩短 | 2064 → 368（5.6x 短） |
| 接受率损失 | 可能 8-12% |
| 参数量 | ~14M |

**适用**：resampler 保留一定的注意力选择能力（比纯 pool 更"聪明"地选哪些信息保留），同时砍掉一半计算。

---

### 方案 C：轻量 + 补信号

```json
{
  "compression_mode": "mean_pool",
  "budget": 64,
  "mask_fusion": "global_summary"
}
```

**架构**：
```
vision tokens [1, 1760, H]
    ↓ adaptive_avg_pool1d(1760 → 64) + Linear + RMSNorm
compressed [1, 64, H]
    ↓ 拼回 → KV 从 2064 缩到 368
    ↓ fusion: mean(compressed) → Linear → gate × 加到 noise_embedding
```

| 指标 | 预期 |
|---|---|
| adapter 计算 | 很小（pool + 2 个 Linear） |
| 压缩比 | 1760→64 = **27.5x** |
| KV 缩短 | 2064 → 368（5.6x 短） |
| 接受率损失 | 可能 8-15%（pool 粗糙但 fusion 补偿一部分） |
| 参数量 | ~50M（2 个 Linear） |

**适用**：pool 压缩快但丢信息多，用 fusion 把全局视觉摘要补回给 draft 预测位置，部分挽回接受率。

---

## 对比总结

| 方案 | 压缩方式 | budget | fusion | adapter 开销 | 预期接受率损失 | 预期 KV 压缩 |
|---|---|---|---|---|---|---|
| **当前** | resampler×2 | 128 | global_summary | 重 | -6% | 1.8x |
| **A 极致轻量** | mean_pool | 32 | off | **几乎为 0** | -15~25% | **6.1x** |
| **B 折中** | resampler×1 | 64 | off | 轻 | -8~12% | 5.6x |
| **C 轻量+补信号** | mean_pool | 64 | global_summary | 很轻 | -8~15% | 5.6x |

---

## 如何修改

只需改 `configs/qwen2.5-vl-7b-dflash-video-adapter.json` 中的 `video_adapter` 字段：

```json
"video_adapter": {
  "enabled": true,
  "compression_mode": "mean_pool",     // 或 "perceiver_resampler"
  "budget": 32,                         // 32 / 64 / 128
  "kv_mode": "draft_only",
  "mask_fusion": "off",                 // "off" / "global_summary"
  "image_token_id": 151655,
  "num_resampler_layers": 1,            // 仅 perceiver_resampler 用
  "num_resampler_heads": 28,
  "num_resampler_kv_heads": 4,
  "num_anchor_raw": 0,
  "gate_init": 0.0
}
```

改完后用 `--freeze-draft-backbone` 重新训 adapter（backbone 已收敛，只需训新 adapter 几千 step），再用 `eval_dflash_spec_video.py` 对比 ON/OFF。

---

## 建议实验路径

1. **先试方案 A**（最快验证极限）：如果 accept_len 从 1.90 掉到 <1.2，说明 budget=32 太激进
2. **再试方案 B**（最可能成功）：1 层 resampler + budget=64，计算量比当前少 75%，接受率预计只掉 8-12%
3. 如果 B 的 speedup 仍 <1，说明**单 batch pytorch 评测天花板就是 adapter 带不来加速**（需要 batched serving 场景才能体现 KV cache 节省的价值）
