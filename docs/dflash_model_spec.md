# DFlash Draft Model 规格说明

## 1. 概述

DFlash 是一个用于 speculative decoding 的 draft model，基于 Qwen3 Transformer 架构，利用 target model 中间层 hidden states 作为上下文，通过 block-wise 并行方式一次预测多个 token。

核心特点：
- 从 target model 的多个中间层提取 hidden states，经 fc 投影后作为 cross-attention 的 KV context
- 每次从一个 anchor token 出发，并行预测一个 block（默认 16 token）
- 训练时随机采样多个 anchor 位置进行并行训练
- 推理时使用 speculative decoding 循环：draft 预测 → target 验证 → 接受/拒绝

---

## 2. 模型配置对比

| 配置项 | Qwen3-8B-DFlash | Qwen3.5-35B-A3B-DFlash | LongCat-Flash-DFlash |
|--------|-----------------|------------------------|---------------------|
| `num_hidden_layers` (draft 层数) | 5 | 8 | 5 |
| `num_target_layers` (target 总层数) | 36 | 40 | 28 |
| `hidden_size` | 4096 | 2048 | 6144 |
| `intermediate_size` | 12288 | 6144 | 12288 |
| `num_attention_heads` | 32 | 32 | 32 |
| `num_key_value_heads` (GQA) | 8 | 4 | 8 |
| `head_dim` | 128 | 128 | 128 |
| `block_size` | 16 | 16 | 16 |
| `target_layer_ids` | [1, 9, 17, 25, 33] | [1, 10, 19, 28, 37] | [1, 7, 13, 19, 25] |
| `vocab_size` | 151936 | 248320 | 131072 |
| `max_position_embeddings` | 40960 | 262144 | 40960 |
| `rope_theta` | 1,000,000 | 10,000,000 | 1,000,000 |
| `model_type` | qwen3 | qwen3 | qwen3 |
| `dtype` | bfloat16 | bfloat16 | bfloat16 |
| `hidden_act` | silu | silu | silu |
| `attention_bias` | false | false | false |

---

## 3. 参数量估算

### 通用公式

Draft model 可训练参数 = fc 投影层 + L 层 Decoder + final norm

每层 Decoder 参数：
- `q_proj`: H × (n_heads × head_dim)
- `k_proj`: H × (n_kv_heads × head_dim)
- `v_proj`: H × (n_kv_heads × head_dim)
- `o_proj`: (n_heads × head_dim) × H
- `gate_proj`: H × intermediate_size
- `up_proj`: H × intermediate_size
- `down_proj`: intermediate_size × H
- `q_norm`, `k_norm`: 2 × head_dim
- `input_layernorm`, `post_attention_layernorm`: 2 × H

fc 投影层：M × H × H （M = len(target_layer_ids)）

### 各模型参数量

#### Qwen3-8B-DFlash (H=4096, L=5, M=5)

| 模块 | 参数量 |
|------|--------|
| fc (5×4096 → 4096) | 83.9M |
| hidden_norm | 4K |
| 每层 attention (q/k/v/o_proj) | 4096×4096 + 4096×1024×2 + 4096×4096 = 41.9M |
| 每层 MLP (gate/up/down_proj) | 4096×12288×3 = 150.9M |
| 每层 norms + q/k_norm | ~12K |
| **每层小计** | **~192.9M** |
| **5层总计** | **~964.5M** |
| final norm | 4K |
| **模型总参数** | **~1,048.5M (~1.05B)** |

#### Qwen3.5-35B-A3B-DFlash (H=2048, L=8, M=5)

| 模块 | 参数量 |
|------|--------|
| fc (5×2048 → 2048) | 21.0M |
| 每层 attention | 2048×4096 + 2048×512×2 + 4096×2048 = 18.9M |
| 每层 MLP | 2048×6144×3 = 37.7M |
| **每层小计** | **~56.6M** |
| **8层总计** | **~452.9M** |
| **模型总参数** | **~474.0M (~0.47B)** |

#### LongCat-Flash-DFlash (H=6144, L=5, M=5)

| 模块 | 参数量 |
|------|--------|
| fc (5×6144 → 6144) | 188.7M |
| 每层 attention | 6144×4096 + 6144×1024×2 + 4096×6144 = 62.9M |
| 每层 MLP | 6144×12288×3 = 226.5M |
| **每层小计** | **~289.4M** |
| **5层总计** | **~1,447M** |
| **模型总参数** | **~1,636M (~1.64B)** |

> 注意：`embed_tokens` 和 `lm_head` 来自 target model，冻结不参与训练，不计入 draft 参数。

---

## 4. 输入输出规格

### 4.1 训练阶段 (OnlineDFlashModel)

**输入：**

| 参数 | 维度 | 说明 |
|------|------|------|
| `input_ids` | `[B, S]` | 原始 token IDs |
| `hidden_states` | `[B, S, M×H]` | Target model M 个中间层 hidden states 拼接 |
| `loss_mask` | `[B, S]` | 标记哪些位置参与 loss 计算 |

**内部处理后的 DFlashDraftModel 输入：**

| 参数 | 维度 | 说明 |
|------|------|------|
| `position_ids` | `[B, S + N×K]` | 完整位置编码（context + draft blocks） |
| `noise_embedding` | `[B, N×K, H]` | 每 block 首 token 为真实 embedding，其余为 MASK |
| `target_hidden` | `[B, S, M×H]` | 经 fc+norm 投影前的 target hidden states |
| `attention_mask` | BlockMask 或 `[B, 1, N×K, S+N×K]` | Block-wise attention mask |

**输出：**

| 参数 | 维度 | 说明 |
|------|------|------|
| `loss` | scalar | 加权 cross-entropy loss |
| `accuracy` | scalar | 预测准确率 |

---

### 4.2 推理阶段 (spec_generate)

**输入：**

| 参数 | 维度 | 说明 |
|------|------|------|
| `input_ids` | `[1, S]` | prompt token IDs |
| `max_new_tokens` | int | 最大生成 token 数 |
| `stop_token_ids` | list[int] | 停止 token |
| `temperature` | float | 采样温度（0 = greedy） |

**推理循环每步：**

| 阶段 | 输入 | 输出 |
|------|------|------|
| Prefill | `input_ids [1, S]` | `hidden_states`, first token |
| Draft block 构造 | `[accepted_token, MASK×(K-1)]` | `noise_embedding [1, K, H]` |
| Draft 预测 | noise_embedding + target_hidden | `draft_logits [1, K-1, V]` → K-1 个候选 token |
| Target 验证 | `[accepted + K-1 draft tokens]` | `posterior [1, K, V]` |
| 接受判定 | draft vs posterior 逐位比对 | acceptance_length (0 ~ K-1) |

**最终输出：**

| 参数 | 维度 | 说明 |
|------|------|------|
| `output_ids` | `[1, S + generated]` | 完整序列（prompt + 生成的 token） |

---

## 5. 模型结构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    DFlashDraftModel                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  target_hidden [B, S, M×H]                                      │
│       │                                                         │
│       ├── fc: Linear(M×H → H, bias=False)                       │
│       └── hidden_norm: RMSNorm(H)                               │
│               │                                                 │
│               ▼                                                 │
│       context [B, S, H]  ──────────────────────┐                │
│                                                │                │
│  noise_embedding [B, N×K, H]                   │                │
│       │                                        │                │
│       ▼                                        ▼                │
│  ┌─── Qwen3DFlashDecoderLayer ×L ────────────────────┐          │
│  │                                                    │          │
│  │  input_layernorm (RMSNorm)                         │          │
│  │       │                                            │          │
│  │  Qwen3DFlashAttention:                             │          │
│  │    Q ← hidden_states (draft)                       │          │
│  │    K ← concat(context, hidden_states)              │          │
│  │    V ← concat(context, hidden_states)              │          │
│  │    + RoPE, GQA, per-head RMSNorm                   │          │
│  │       │                                            │          │
│  │  + residual                                        │          │
│  │       │                                            │          │
│  │  post_attention_layernorm (RMSNorm)                │          │
│  │       │                                            │          │
│  │  Qwen3MLP (SwiGLU: gate_proj + up_proj → down_proj)│          │
│  │       │                                            │          │
│  │  + residual                                        │          │
│  └────────────────────────────────────────────────────┘          │
│       │                                                         │
│       ▼                                                         │
│  norm: RMSNorm(H)                                               │
│       │                                                         │
│       ▼                                                         │
│  output [B, N×K, H]                                             │
│       │                                                         │
│       ▼ (接 target lm_head, 冻结)                                │
│  logits [B, N×K, V]                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Attention Mask 设计

DFlash 的 attention mask 是其核心设计，决定了信息流方向：

```
KV 维度: [context_0, ..., context_{S-1}, block_0_tok_0, ..., block_0_tok_{K-1}, block_1_tok_0, ...]
Q 维度:  [block_0_tok_0, ..., block_0_tok_{K-1}, block_1_tok_0, ...]

规则:
1. block_i 的 Q 可以 attend to context 中 anchor_pos_i 之前的所有 token
2. block_i 内部 token 之间双向可见（非 causal）
3. 不同 block 之间完全隔离
4. 无效 block (keep_mask=False) 不 attend to 任何位置
```

支持两种 backend：
- `flex_attention`：PyTorch 2.x 的 BlockMask，编译时优化
- `sdpa`：显式 bool tensor mask `[B, 1, N×K, S+N×K]`

---

## 7. 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_anchors` | 512 | 每个序列采样的 anchor 数量 |
| `block_size` | 16 | 每个 anchor 向后预测的 token 数 |
| `attention_backend` | flex_attention | attention mask 实现方式 |
| `loss_decay_gamma` | None (可选 7.0) | 指数衰减系数，远处位置权重降低 |
| `mask_token_id` | 模型特定 | MASK token 的 ID |

---

## 8. 关键设计决策

1. **fc 投影层**：将 M 个 target 层的 hidden states（维度 M×H）压缩到 H，而非逐层独立处理，节省计算。

2. **Block 内双向 attention**：不同于 autoregressive 的 causal mask，block 内部 token 互相可见，允许更好的并行预测。

3. **Cross + Self Attention 混合**：K/V 同时包含 target context 和 draft 自身 hidden states，让模型既能参考 target 信息又能利用自身预测。

4. **共享 embed_tokens/lm_head**：复用 target model 的 embedding 和输出层，大幅减少参数量且保证 token 空间一致。

5. **target_layer_ids 均匀分布**：从 target 模型的不同深度均匀采样中间层，捕获从浅层语法到深层语义的多级特征。
