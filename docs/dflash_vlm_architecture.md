# DFlash VLM 架构全流程

## 一、整体概念

DFlash 是一种**块并行投机解码**方案：一个轻量 draft 模型基于 target 模型的 hidden states，一次预测一个 block（16 token）的候选 token，再由 target 验证接受/拒绝。

```
┌──────────────────────────────────────────────────────────┐
│                      训练阶段                              │
│                                                          │
│  input_ids + image_files                                 │
│       │                                                  │
│       ▼                                                  │
│  ┌─────────────┐     target hidden      ┌────────────┐  │
│  │ Target Model│ ──────────────────────► │ Draft Model│  │
│  │ (冻结,7B/35B)│    [1, S, H×3层]       │ (训练,轻量) │  │
│  └─────────────┘                         └────────────┘  │
│       │                                       │          │
│       │ 无梯度                                 │ 有梯度    │
│       ▼                                       ▼          │
│   pixel_values                          预测 block 内     │
│   + image_grid_thw                      每个位置的 token   │
│   + hidden states 提取                  → CE loss         │
└──────────────────────────────────────────────────────────┘
```

---

## 二、训练数据流（完整链路）

```
JSONL 样本
  {"input_ids": [...], "loss_mask": [...], "image_files": ["f0.jpg",...,"f7.jpg"]}
         │
         ▼
┌─── OnlineMMDataset ───┐
│  归一: image_files     │
│  截断: max_len=4096    │
└────────────┬──────────┘
             │
             ▼
┌─── OnlineMMCollator ──┐
│  升维 [1,S]            │
│  透传 image_files      │
└────────────┬──────────┘
             │
             ▼
┌─────── 训练循环 ──────────────────────────────────────────────────┐
│                                                                   │
│  ① Target Forward (HFDFlashTargetModel.generate_dflash_data)     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ images = [Image.open(p) for p in image_files]               │ │
│  │ img_proc = processor.image_processor(images=images)         │ │
│  │   → pixel_values [N_patches, D]                             │ │
│  │   → image_grid_thw [N_frames, 3]                           │ │
│  │                                                             │ │
│  │ 校验断言: grid.prod//merge²之和 == input_ids中image_token数  │ │
│  │                                                             │ │
│  │ model.forward(input_ids, pixel_values, image_grid_thw,      │ │
│  │              output_hidden_states=True)                      │ │
│  │   → hidden_states[layer1] ‖ [layer13] ‖ [layer24]          │ │
│  │   → 拼接 → [1, S, H×3] (如 3584×3 = 10752)                │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                          │                                        │
│                          ▼                                        │
│  ② Draft Forward (OnlineDFlashModel.forward)                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  target_hidden [1, S, 10752]                                │ │
│  │       │                                                     │ │
│  │       ▼                                                     │ │
│  │  ┌── fc + hidden_norm ──┐                                   │ │
│  │  │  Linear(10752→H) + RMSNorm  → [1, S, H]                 │ │
│  │  └──────────┬───────────┘                                   │ │
│  │             │                                               │ │
│  │             ▼                                               │ │
│  │  ┌── Video Adapter (可选) ──────────────────────────┐       │ │
│  │  │  compress: Nv个vision token → budget个            │       │ │
│  │  │  fuse_mask: 视觉摘要注入noise_embedding           │       │ │
│  │  │  (详见第三节)                                     │       │ │
│  │  └──────────┬───────────────────────────────────────┘       │ │
│  │             │                                               │ │
│  │             ▼                                               │ │
│  │  ┌── Anchor Sampling ──┐                                    │ │
│  │  │  从 loss_mask=1 区域随机采 num_anchors(512) 个位置        │ │
│  │  │  每个 anchor 展开一个 block(16 token) 的预测窗口          │ │
│  │  └──────────┬───────────┘                                   │ │
│  │             │                                               │ │
│  │             ▼                                               │ │
│  │  ┌── Noise Embedding ──┐                                    │ │
│  │  │  block 内每个位置: embed(mask_token) + position_embed     │ │
│  │  │  (+ fusion from video adapter if enabled)                │ │
│  │  └──────────┬───────────┘                                   │ │
│  │             │                                               │ │
│  │             ▼                                               │ │
│  │  ┌── Draft Transformer Layers ──┐                           │ │
│  │  │  input = [context ‖ noise_embedding]                     │ │
│  │  │  context = target_hidden (压缩后 or 原始)                │ │
│  │  │  每层: self-attn(causal + block mask) + cross 看 context │ │
│  │  │  output → [1, N_anchors × block_size, H]                 │ │
│  │  └──────────┬───────────┘                                   │ │
│  │             │                                               │ │
│  │             ▼                                               │ │
│  │  ┌── Projector (预测头) ──┐                                 │ │
│  │  │  LM head: Linear(H → vocab_size)                         │ │
│  │  │  或 GRU projector                                        │ │
│  │  │  → logits [1, N_anchors × block_size, vocab]             │ │
│  │  └──────────┬───────────┘                                   │ │
│  │             │                                               │ │
│  │             ▼                                               │ │
│  │  ┌── Loss ──┐                                               │ │
│  │  │  CE(logits, ground_truth_token_at_each_position)         │ │
│  │  │  + loss_decay (gamma=7, block 内越远权重越低)             │ │
│  │  └──────────┘                                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

---

## 三、Video Adapter 详细流程

当 draft config 启用 `"video_adapter"` 时，在 draft forward 内部插入压缩+融合：

```
target_hidden (fc+norm 后) [1, S, H]
  其中 S = Nt(文本token) + Nv(视觉token, 如8帧×220=1760)
         │
         ▼
┌─── compress() ─────────────────────────────────────────────────────┐
│                                                                     │
│  ① 分离: visual_tokens = target_hidden[:, image_token位置, :]       │
│          → [1, Nv, H]  (如 [1, 1760, 3584])                        │
│                                                                     │
│  ② 压缩 (三选一):                                                   │
│                                                                     │
│     ┌─ perceiver_resampler ─────────────────────────┐               │
│     │  learnable queries [1, budget, H]             │               │
│     │       ↓ cross-attention (Q=queries, KV=visual)│               │
│     │       ↓ × num_layers (2层)                    │               │
│     │       ↓ RMSNorm                               │               │
│     │  → compressed [1, budget, H]                  │               │
│     │    (如 budget=64: 1760→64, 压缩27.5x)         │               │
│     └───────────────────────────────────────────────┘               │
│                                                                     │
│     ┌─ mean_pool ───────────────────────────────────┐               │
│     │  adaptive_avg_pool1d(Nv → budget)             │               │
│     │  + Linear + RMSNorm                           │               │
│     │  → compressed [1, budget, H]                  │               │
│     └───────────────────────────────────────────────┘               │
│                                                                     │
│     ┌─ hybrid ──────────────────────────────────────┐               │
│     │  resampled = perceiver_resampler(visual)      │               │
│     │  anchors = top-k by L2 norm (关键帧token)      │               │
│     │  compressed = cat([resampled, anchors])       │               │
│     │  → [1, budget + K, H]                         │               │
│     └───────────────────────────────────────────────┘               │
│                                                                     │
│  ③ 拼回完整序列:                                                     │
│     text_before [vis_start] + compressed [budget] + text_after      │
│     → compressed_target_hidden [1, S_compressed, H]                 │
│       (原 S=2064 → S_compressed=2064-1760+64=368)                   │
│                                                                     │
│  返回: (compressed_target_hidden, compressed_visual, info)           │
└─────────────────────────────────────────────────────────────────────┘
         │                              │
         │ 去处①: draft KV context       │ 去处②: fuse 进 noise embedding
         ▼                              ▼
┌── draft attention ──┐    ┌─── fuse_mask() ───────────────────────┐
│ context 只有 368 长   │    │                                       │
│ (vs 原来 2064)        │    │  global_summary:                      │
│ attention 计算省 5.6x │    │    mean(compressed) → [1,1,H]         │
│                      │    │    + gate * proj(mean) 加到 noise      │
│                      │    │                                       │
│                      │    │  mask_cross_attention:                 │
│                      │    │    cross_attn(noise, compressed)       │
│                      │    │    + gate * attn_out 加到 noise        │
│                      │    │                                       │
│                      │    │  gate 初始=0, 训练中慢慢学开            │
└──────────────────────┘    └───────────────────────────────────────┘
```

---

## 四、推理阶段（spec_generate）

```
input_ids (prompt + 已生成 token)
         │
         ▼
┌─── Target Prefill ────────────────────────────────────┐
│  target.forward(input_ids, use_cache=True)             │
│  → KV cache + hidden_states 多层拼接                   │
│  → 采样第一个 token                                    │
│                                                        │
│  Video Adapter: 压缩 prefill 的 target_hidden          │
│  → draft 的初始 KV context 已是压缩版本                 │
└────────────────────────┬──────────────────────────────┘
                         │
                         ▼
┌─── Decode Loop ────────────────────────────────────────┐
│                                                        │
│  repeat until EOS or max_length:                       │
│                                                        │
│    ① Draft 一次性预测 block_size(16) 个候选 token       │
│       [用压缩后的 context KV + 当前位置]                │
│       → candidates [16 tokens]                         │
│                                                        │
│    ② Target 验证: 一次 forward 校验 16 个 token         │
│       → 找到第一个不匹配的位置 k                        │
│       → 接受前 k 个, 拒绝后 16-k 个                    │
│       → 更新 KV cache                                  │
│                                                        │
│    acceptance_length += k                              │
│                                                        │
└────────────────────────────────────────────────────────┘

加速比 = total_tokens / (prefill_steps + decode_steps)
接受率 = mean(acceptance_length) / block_size
```

---

## 五、关键模块对应文件

| 模块 | 文件 | 核心函数/类 |
|---|---|---|
| Target 模型 forward | `specforge/modeling/target/dflash_target_model.py` | `HFDFlashTargetModel.generate_dflash_data` |
| Draft 模型结构 | `specforge/modeling/draft/dflash.py` | `DFlashDraftModel.forward` |
| Video Adapter | `specforge/modeling/draft/video_adapter.py` | `ElasticVideoAdapter` |
| 训练编排(anchor/mask/loss) | `specforge/core/dflash.py` | `OnlineDFlashModel.forward` |
| 数据集 | `specforge/data/online_mm_dataset.py` | `OnlineMMDataset` / `OnlineMMCollator` |
| 训练脚本 | `scripts/train_dflash.py` | `main()` |
| 数据生成(视频) | `scripts/gen_video_dflash_data.py` | `main()` |

---

## 六、视频场景的特殊处理

| 问题 | 解法 |
|---|---|
| 视频 = 多帧图片(非原生 video) | 评测端用 opencv 抽帧成 N 个 image_url，训练同理 |
| 多帧 vision token 太多(8帧=1760) | Video Adapter 压缩到 budget(64)，减少 draft 计算 |
| Draft 用 1D position(非 3D mrope) | 多帧当多图处理，只用 image_token，不引入 temporal 维 |
| Pixels 必须一致 | 生成数据/训练/评测三端 min_pixels/max_pixels 必须相同 |
| Loss 只在生成段算 | loss_mask=0 的 prompt+vision token 位置不参与 loss |
