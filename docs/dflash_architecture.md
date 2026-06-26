# DFlash Draft Model 架构详解

## 概述

DFlash 是一个基于 Qwen3 架构的 speculative decoding draft model，核心思想是利用 target model 中间层的 hidden states 作为上下文，通过 block-wise 并行预测多个 token。训练时从序列中随机采样多个 anchor 位置，每个 anchor 向后预测一个 block（默认 16 个 token）。

---

## 整体训练流程 (OnlineDFlashModel)

```
Target Model (frozen)
    │
    ├── embed_tokens: 提供 noise embedding
    ├── hidden_states: 提供中间层特征作为 context
    └── lm_head: 将 draft output 映射回 vocab
    
DFlash Draft Model (trainable)
    │
    ├── fc + hidden_norm: 投影 target hidden states
    ├── Qwen3DFlashDecoderLayer × N: Transformer layers
    └── norm: 最终 RMSNorm
```

---

## 维度符号定义

| 符号 | 含义 | 典型值 |
|------|------|--------|
| B | batch size | 1 |
| S | 序列长度 (seq_len) | 3072 |
| H | hidden_size | 3584 (Qwen3-4B) |
| V | vocab_size | 151936 |
| N | num_anchors (采样的 anchor 数量) | 512 |
| K | block_size (每个 anchor 预测的 token 数) | 16 |
| L | num_draft_layers | 1 |
| T | num_target_layers (target model 总层数) | 36 (Qwen3-4B) |
| M | 捕获的 target 层数 (len(target_layer_ids)) | 等于 L (默认 1) |
| n_heads | num_attention_heads | 32 |
| n_kv_heads | num_key_value_heads | 8 |
| head_dim | 每个 head 的维度 | 128 |

---

## 模块详细架构

### 1. Target Model 输出 (DFlashTargetModel)

**输入:**
- `input_ids`: `[B, S]` — 原始 token IDs
- `attention_mask`: `[B, S]`
- `loss_mask`: `[B, S]`

**输出 (DFlashTargetOutput):**
- `hidden_states`: `[B, S, M * H]` — 拼接 M 个中间层的 hidden states
  - 例如 M=1 时为 `[B, S, H]`，M=2 时为 `[B, S, 2*H]`

**target_layer_ids 选择逻辑:**
- 当 `num_draft_layers=1` 时: 选取 target 模型正中间层 `[T//2]`
- 当 `num_draft_layers>1` 时: 均匀分布在第 1 层到第 T-3 层之间

---

### 2. Anchor 采样 (_sample_anchor_positions)

**输入:**
- `loss_mask`: `[B, S]`

**输出:**
- `anchor_positions`: `[B, N]` — 每个样本采样 N 个有效位置 (值域 [0, S-K])
- `block_keep_mask`: `[B, N]` — bool，标记每个 anchor 是否有效

---

### 3. Noise Embedding 构造 (_create_noise_embed)

每个 block 中第一个位置放入真实 token 的 embedding，其余 K-1 个位置填入 MASK token embedding。

**输入:**
- `input_ids`: `[B, S]`
- `anchor_positions`: `[B, N]`
- `block_keep_mask`: `[B, N]`

**处理:**
1. 创建 noise_ids: `[B, N*K]`，全部填 MASK token ID
2. 每个 block 的第 0 位替换为 `input_ids[anchor_pos]`
3. 通过 target embed_tokens 嵌入

**输出:**
- `noise_embedding`: `[B, N*K, H]`

---

### 4. Position IDs 构造 (_create_position_ids)

**Context 部分:** `[0, 1, 2, ..., S-1]` → `[B, S]`

**Draft 部分:** 对每个 anchor_pos，生成 `[anchor_pos, anchor_pos+1, ..., anchor_pos+K-1]` → `[B, N*K]`

**完整 position_ids:** `[B, S + N*K]` — 拼接 context 和 draft 部分

---

### 5. Attention Mask 构造

KV 布局: `[Context (S tokens) | Block_0 (K tokens) | Block_1 (K tokens) | ... | Block_{N-1} (K tokens)]`

Q 布局: `[Block_0 (K tokens) | Block_1 (K tokens) | ... | Block_{N-1} (K tokens)]`

**Mask 规则:**
1. 每个 block 可以看到 context 中 anchor_pos 之前的所有 token (kv_idx < anchor_pos)
2. 同一 block 内的 token 互相可见（双向 attention）
3. 不同 block 之间互不可见
4. 无效 block (block_keep_mask=False) 什么都看不到

**Mask 维度:**
- flex_attention 模式: `BlockMask` 对象，逻辑维度 Q_LEN=N*K, KV_LEN=S+N*K
- sdpa 模式: `[B, 1, N*K, S+N*K]` bool tensor

---

### 6. DFlashDraftModel Forward

#### 6.1 输入

| 参数 | 维度 | 说明 |
|------|------|------|
| `position_ids` | `[B, S + N*K]` | 完整位置编码（含 context 和 draft） |
| `noise_embedding` | `[B, N*K, H]` | Draft tokens 的嵌入 |
| `target_hidden` | `[B, S, M*H]` | Target 模型中间层拼接特征 |
| `attention_mask` | BlockMask 或 `[B, 1, N*K, S+N*K]` | Attention mask |

#### 6.2 fc 投影层

将多层 target hidden states 投影到 draft model 的 hidden_size：

```
target_hidden: [B, S, M*H] → fc → [B, S, H] → hidden_norm (RMSNorm) → [B, S, H]
```

- `self.fc`: `nn.Linear(M * H, H, bias=False)`
- `self.hidden_norm`: `Qwen3RMSNorm(H)`

#### 6.3 Rotary Position Embedding

```
position_embeddings = self.rotary_emb(hidden_states, position_ids)
```

- 输入 `position_ids`: `[B, S + N*K]`
- 输出 `(cos, sin)`: 各 `[B, S + N*K, head_dim]`

注意: cos/sin 覆盖 context + draft 的所有位置，在 attention 中 Q 用 draft 部分的位置，K 用全部位置。

#### 6.4 Qwen3DFlashDecoderLayer (× L 层)

每一层结构：

```
                   hidden_states [B, N*K, H]
                         │
                 ┌───────┴───────┐
                 │  input_layernorm (RMSNorm)
                 │       │
                 │  Qwen3DFlashAttention
                 │       │
                 └───+───┘  (residual connection)
                     │
                     │ [B, N*K, H]
                 ┌───┴───┐
                 │  post_attention_layernorm (RMSNorm)
                 │       │
                 │  Qwen3MLP (SwiGLU)
                 │       │
                 └───+───┘  (residual connection)
                     │
                     │ [B, N*K, H]
```

#### 6.5 Qwen3DFlashAttention 详细

这是 DFlash 的核心创新 — cross-attention + self-attention 混合：

**Q 来源:** `hidden_states` (draft tokens)
**K/V 来源:** `concat([target_hidden, hidden_states])` (context + draft)

```
Q = q_proj(hidden_states)           → [B, N*K, n_heads * head_dim]
    reshape                         → [B, N*K, n_heads, head_dim]
    q_norm (RMSNorm per head)       → [B, N*K, n_heads, head_dim]
    transpose(1,2)                  → [B, n_heads, N*K, head_dim]

K_ctx = k_proj(target_hidden)       → [B, S, n_kv_heads * head_dim]
K_noise = k_proj(hidden_states)     → [B, N*K, n_kv_heads * head_dim]
K = concat([K_ctx, K_noise], dim=1) → [B, S + N*K, n_kv_heads * head_dim]
    reshape                         → [B, S + N*K, n_kv_heads, head_dim]
    k_norm (RMSNorm per head)       → [B, S + N*K, n_kv_heads, head_dim]
    transpose(1,2)                  → [B, n_kv_heads, S + N*K, head_dim]

V_ctx = v_proj(target_hidden)       → [B, S, n_kv_heads * head_dim]
V_noise = v_proj(hidden_states)     → [B, N*K, n_kv_heads * head_dim]
V = concat([V_ctx, V_noise], dim=1) → [B, S + N*K, n_kv_heads, head_dim]
    transpose(1,2)                  → [B, n_kv_heads, S + N*K, head_dim]

# RoPE
Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

# Attention (with GQA: n_heads / n_kv_heads groups)
attn_output = attention(Q, K, V, mask)  → [B, n_heads, N*K, head_dim]
    reshape                              → [B, N*K, n_heads * head_dim]

output = o_proj(attn_output)             → [B, N*K, H]
```

**线性层参数:**
| 层 | 输入维度 | 输出维度 |
|---|---|---|
| q_proj | H | n_heads * head_dim |
| k_proj | H | n_kv_heads * head_dim |
| v_proj | H | n_kv_heads * head_dim |
| o_proj | n_heads * head_dim | H |

#### 6.6 Qwen3MLP (SwiGLU)

```
gate = gate_proj(x)     → [B, N*K, intermediate_size]
up = up_proj(x)         → [B, N*K, intermediate_size]
hidden = silu(gate) * up → [B, N*K, intermediate_size]
output = down_proj(hidden) → [B, N*K, H]
```

| 层 | 输入维度 | 输出维度 |
|---|---|---|
| gate_proj | H | intermediate_size |
| up_proj | H | intermediate_size |
| down_proj | intermediate_size | H |

(intermediate_size 对 Qwen3-4B 为 18944)

#### 6.7 Final Norm

```
output = self.norm(hidden_states)  → [B, N*K, H]
```

---

### 7. LM Head (frozen, from target model)

```
logits = lm_head(draft_output)  → [B, N*K, V]
reshape → [B, N, K, V]
```

- `lm_head`: `nn.Linear(H, V, bias=False)` — 权重冻结，来自 target model

---

### 8. Loss 计算

**Labels:** 每个 anchor 位置 p 的 block 预测 `input_ids[p], input_ids[p+1], ..., input_ids[p+K-1]`

**Weight Mask 构成:**
1. `block_keep_mask` — 无效 block 的 loss 为 0
2. `valid_label_mask` — 超出序列边界的位置为 0
3. `pos_in_block > 0` — block 的第 0 位（anchor token 本身）不计入 loss
4. `original_loss_mask` — 原始 loss mask（如 padding 区域不计算）
5. (可选) 指数衰减: `exp(-(k-1)/γ)`，越远的位置权重越低

**最终 loss:**
```
loss = sum(CE(logits, labels) * weight_mask) / sum(weight_mask)
```

---

## 推理流程 (spec_generate)

```
1. Prefill: target model 处理 input_ids → 得到 hidden_states, first token
2. Loop:
   a. 构造 block: [accepted_token, MASK, MASK, ..., MASK] (K tokens)
   b. Draft model 预测: block → draft_logits → sample K-1 个 token
   c. Target model 验证: block → posterior logits
   d. 逐位比对 acceptance，找到最长匹配前缀
   e. 更新 KV cache，移动指针
```

---

## 参数量估算 (以 Qwen3-4B, L=1, M=1 为例)

| 模块 | 参数量 |
|------|--------|
| fc (H → H) | H² = 3584² ≈ 12.8M |
| hidden_norm | H = 3584 |
| Layer 0 - q_proj | H × (n_heads × head_dim) = 3584 × 4096 ≈ 14.7M |
| Layer 0 - k_proj | H × (n_kv_heads × head_dim) = 3584 × 1024 ≈ 3.7M |
| Layer 0 - v_proj | H × (n_kv_heads × head_dim) = 3584 × 1024 ≈ 3.7M |
| Layer 0 - o_proj | (n_heads × head_dim) × H = 4096 × 3584 ≈ 14.7M |
| Layer 0 - q_norm, k_norm | 2 × head_dim = 256 |
| Layer 0 - input_layernorm | H = 3584 |
| Layer 0 - gate_proj | H × intermediate = 3584 × 18944 ≈ 67.9M |
| Layer 0 - up_proj | H × intermediate = 3584 × 18944 ≈ 67.9M |
| Layer 0 - down_proj | intermediate × H = 18944 × 3584 ≈ 67.9M |
| Layer 0 - post_attention_layernorm | H = 3584 |
| norm (final) | H = 3584 |
| rotary_emb | 0 (动态计算) |
| **总计** | **~253M** |

注意: embed_tokens 和 lm_head 来自 target model，冻结不训练。

---

## 数据流总结图

```
input_ids [B, S]
     │
     ├─── target_model ──→ hidden_states [B, S, M*H]
     │                          │
     │                     fc [M*H → H]
     │                          │
     │                     hidden_norm
     │                          │
     │                     target_hidden [B, S, H]  ← 作为 attention 的 context KV
     │
     ├─── sample_anchors ──→ anchor_positions [B, N]
     │                        block_keep_mask [B, N]
     │
     ├─── create_noise_embed ──→ noise_embedding [B, N*K, H]
     │         (embed_tokens)         │
     │                                │ (作为 hidden_states 输入 draft layers)
     │                                ▼
     │                    ┌─── DFlashDecoderLayer ×L ───┐
     │                    │                              │
     │                    │  Q ← hidden_states [B,N*K,H] │
     │                    │  K ← [target_hidden | hidden_states] │
     │                    │      [B, S+N*K, H]           │
     │                    │  V ← [target_hidden | hidden_states] │
     │                    │      [B, S+N*K, H]           │
     │                    │                              │
     │                    │  + MLP (SwiGLU)              │
     │                    └──────────────────────────────┘
     │                                │
     │                           final_norm
     │                                │
     │                         draft_output [B, N*K, H]
     │                                │
     │                           lm_head (frozen)
     │                                │
     │                         logits [B, N*K, V]
     │                                │
     └──── labels (input_ids shifted) │
                                      ▼
                              CE Loss + Accuracy
```
