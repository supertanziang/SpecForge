# DFlash 训练代码与架构详解

## 1. DFlash 是什么

DFlash 是一种**投机解码 (Speculative Decoding)** 的 draft 模型训练方法。核心思想：

- 训练一个**轻量 draft 模型**（~473M 参数），它**只消费 target 模型的中间层 hidden states**（不读图像）
- 推理时 draft 一次**并行猜出 block_size 个 token**，再由 target 模型验证
- 因为 draft 不需要处理图像，所以可以直接挂在 VLM（如 Qwen2.5-VL-7B）上

## 2. 模型架构 (DFlashDraftModel)

文件：`specforge/modeling/draft/dflash.py`

### 2.1 整体结构

```
DFlashDraftModel
├── fc: Linear(num_target_layers * target_hidden_size, hidden_size)  # 多层特征投影
├── hidden_norm: RMSNorm(hidden_size)                                 # 投影后归一化
├── layers: ModuleList[Qwen3DFlashDecoderLayer × num_hidden_layers]   # draft transformer 层
├── norm: RMSNorm(hidden_size)                                        # 输出归一化
└── rotary_emb: Qwen3RotaryEmbedding                                  # RoPE 位置编码
```

### 2.2 配置示例 (Qwen2.5-VL-7B-DFlash)

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_size` | 3584 | 与 target 同维（共享 lm_head） |
| `num_hidden_layers` | 5 | draft 层数（远小于 target 的 28 层） |
| `num_attention_heads` | 28 | |
| `num_key_value_heads` | 4 | GQA |
| `intermediate_size` | 18944 | MLP 中间维度 |
| `head_dim` | 128 | |
| `block_size` | 16 | 每次猜 16 个 token |
| `target_layer_ids` | [1, 7, 13, 19, 25] | 从 target 抽取哪些层的 hidden states |

### 2.3 Qwen3DFlashDecoderLayer（单层结构）

每层与标准 Transformer 类似，但 attention 有关键不同：

```
Input: hidden_states [B, Q_LEN, H], target_hidden [B, S, H]
  │
  ├─ input_layernorm → Qwen3DFlashAttention
  │                      Q = q_proj(hidden_states)     → [B, heads, Q_LEN, head_dim]
  │                      K = [k_proj(target_hidden); k_proj(hidden_states)]  → [B, heads, S+Q_LEN, head_dim]
  │                      V = [v_proj(target_hidden); v_proj(hidden_states)]  → [B, heads, S+Q_LEN, head_dim]
  │                      ↓ RoPE → Attention(Q, K, V, mask) → o_proj
  │
  ├─ + residual
  │
  ├─ post_attention_layernorm → Qwen3MLP (SwiGLU)
  │
  └─ + residual → output [B, Q_LEN, H]
```

**关键设计**：K/V 是 **context（target hidden states）和 draft 自身拼接**的——draft 既能看到 target 的全局信息，又能做块内自注意力。

## 3. 训练流程 (OnlineDFlashModel)

文件：`specforge/core/dflash.py`

### 3.1 一次 forward 的完整数据流

```
输入:
  input_ids:      [B, S]          ← 完整 token 序列 (含 system/user/assistant)
  hidden_states:  [B, S, H*L]     ← target 模型多层 hidden states 拼接
  loss_mask:      [B, S]          ← 哪些位置参与 loss (assistant 部分=1)

步骤:
  1. 采样 anchor 位置
  2. 构造 noise embedding（块输入）
  3. 构造 attention mask
  4. draft 前向
  5. lm_head + cross-entropy loss
```

### 3.2 详细维度追踪

以 `B=1, S=375, block_size=16, num_anchors=512` 为例（实际 N≤512，取决于 valid 位置数）：

#### Step 1: 采样 anchor 位置

```python
anchor_positions, block_keep_mask = _sample_anchor_positions(S, loss_mask, device)
# anchor_positions: [B, N]       如 [1, 300]   ← N 个随机起始位置（从 loss_mask=1 的区域采）
# block_keep_mask:  [B, N]  bool ← 哪些 anchor 有效
```

#### Step 2: 构造 noise embedding

```python
noise_embedding = _create_noise_embed(input_ids, anchor_positions, block_keep_mask)
# noise_embedding: [B, N*block_size, hidden_size] 如 [1, 4800, 3584]
```

构造规则：每个 block 的**第一个 token 是真实 token（anchor 位置的 token）**，其余全是 **MASK token embedding**。

```
Block_i:  [real_token_embed, MASK_embed, MASK_embed, ..., MASK_embed]
              ↑ position anchor_i       ←── block_size-1 个 MASK ──→
```

#### Step 3: target hidden states 投影

```python
# hidden_states 进入 draft 前先经过 fc 投影:
# 输入: [B, S, hidden_size * len(target_layer_ids)]  如 [1, 375, 3584*5=17920]
# 输出: [B, S, hidden_size]                          如 [1, 375, 3584]
target_hidden = self.hidden_norm(self.fc(target_hidden))
```

#### Step 4: 位置编码与 attention mask

```python
# Position IDs:
context_position_ids: [B, S]              如 [1, 375]      ← 0,1,2,...,374
draft_position_ids:   [B, N*block_size]   如 [1, 4800]     ← 每个 block 从 anchor_i 开始连续编号
full_position_ids:    [B, S + N*block_size]                  ← 拼接

# Attention Mask (flex_attention / sdpa):
# Q:  只有 draft 部分 → [N*block_size] 个 query
# KV: [context(S) | draft(N*block_size)] 拼接
# 规则:
#   - Block_i 的 query 可以看到 context 中 position < anchor_i 的所有 token
#   - Block_i 的 query 可以看到 Block_i 自身（块内双向注意力）
#   - Block_i 看不到其他 Block_j (j≠i)
#   - invalid block (keep_mask=False) 什么都看不到
```

#### Step 5: draft 前向

```python
output_hidden = draft_model(
    position_ids=full_position_ids,         # [B, S + N*bs]
    noise_embedding=noise_embedding,        # [B, N*bs, H]
    target_hidden=target_hidden_proj,       # [B, S, H]     (投影后)
    attention_mask=dflash_attn_mask,
)
# output_hidden: [B, N*block_size, hidden_size]  如 [1, 4800, 3584]
```

#### Step 6: Loss 计算

```python
# Labels: position k (block 内) 预测 input_ids[anchor_i + k]
# 只算 k>0 的位置 (k=0 是 anchor 本身，不需要预测)
# weight_mask 综合: block有效 × 位置合法 × k>0 × loss_mask × 指数衰减

# lm_head 分块计算 (避免一次性分配 N*bs × vocab_size 的巨大 tensor):
chunk_logits = lm_head(output_hidden_chunk)  # [chunk, vocab_size] 如 [512, 152064]
loss = cross_entropy(chunk_logits, targets, reduction='none') * weights

# 最终输出:
loss:     scalar (加权平均 CE)
accuracy: scalar (argmax 命中率，不含 k=0)
```

### 3.3 Loss 衰减 (loss_decay_gamma)

块内越靠后的位置越难预测，用指数衰减降低远端 loss 的权重：

```
weight(k) = exp(-(k-1) / gamma)     gamma=7.0 时:
  k=1: 1.000  (第一个预测位置，权重最大)
  k=2: 0.867
  k=5: 0.565
  k=10: 0.279
  k=15: 0.136
```

## 4. 训练数据管线 (train_dflash.py)

```
每个 training step:
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. 从 dataloader 取 batch:                                   │
  │    input_ids [B, S], attention_mask [B, S], loss_mask [B, S] │
  │    (+ image_file if VLM)                                     │
  │                                                              │
  │ 2. Target model forward (frozen):                            │
  │    hidden_states = target_model.generate_dflash_data(...)    │
  │    → DFlashTargetOutput.hidden_states: [B, S, H*L]           │
  │    (L = len(target_layer_ids), 如 5 层 × 3584 = 17920)       │
  │                                                              │
  │ 3. Draft model forward + loss (可训练):                       │
  │    loss, acc = dflash_model(input_ids, hidden_states, mask)  │
  │                                                              │
  │ 4. loss.backward() → optimizer.step()                        │
  └─────────────────────────────────────────────────────────────┘
```

Target model 有两种 backend：
- **hf**：直接 `AutoModelForCausalLM` 跑 forward，抽取指定层 hidden states
- **sglang**：用 sglang 的 `SGLangRunner` 做 forward，适合大模型 + 量化

## 5. 推理流程 (spec_generate)

文件：`specforge/modeling/draft/dflash.py` → `DFlashDraftModel.spec_generate()`

```
循环:
  1. draft 拿最近的 target hidden + 当前 token，一次猜 block_size 个
  2. target verify：逐个检查 draft 猜的对不对
  3. 接受连续命中的前缀 (acceptance_length)
  4. target 的最后一个验证位置的 token 作为下一轮起点
  5. 循环直到 max_tokens 或遇到 stop token
```

## 6. 关键超参数总结

| 超参数 | 含义 | 典型值 |
|--------|------|--------|
| `block_size` | 每次投机猜测的 token 数 | 16 |
| `num_anchors` | 每条训练样本采样多少个 block 起始位置 | 512 |
| `target_layer_ids` | 从 target 抽取哪些层的 hidden states | [1,7,13,19,25] |
| `num_hidden_layers` | draft 模型层数 | 5 |
| `loss_decay_gamma` | 块内远端 loss 衰减系数 | 7.0 |
| `mask_token_id` | MASK token 的 id（VLM 须手动指定空槽） | 151657 / 248070 |
| `hidden_size` | draft 隐层维度（须与 target 对齐以共享 lm_head） | 3584 / 2048 |

## 7. 为什么 DFlash 适合 VLM

- Draft 只消费 target 的 1D hidden states（文本 token 维度），**不读图像**
- Target 做 forward 时处理图像 → 产生 hidden states → draft 利用这些蒸馏信息猜 token
- 这意味着 draft 模型是纯文本结构，非常轻量，且不需要图像编码器

## 8. 文件索引

| 文件 | 职责 |
|------|------|
| `specforge/modeling/draft/dflash.py` | DFlashDraftModel 定义（网络结构 + 推理） |
| `specforge/core/dflash.py` | OnlineDFlashModel（训练 wrapper：采样、mask、loss） |
| `scripts/train_dflash.py` | 训练入口脚本（数据加载、FSDP、优化器、训练循环） |
| `configs/qwen2.5-vl-7b-dflash.json` | 7B VLM 的 draft 配置 |
| `configs/qwen3.5-35b-a3b-dflash.json` | 35B MoE 的 draft 配置 |
| `specforge/modeling/target/dflash_target_model.py` | Target model 抽象（hf/sglang backend） |
