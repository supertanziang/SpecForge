# Eagle3 VLM 训练流程 (Qwen2.5-VL-7B, 1D RoPE)

本文档记录 SpecForge eagle3 对 Qwen2.5-VL-7B 做投机解码 draft 训练的完整流程。

> **重要改动 (2026-06-16)**：draft 从 3D mrope 改为 **1D RoPE**，对齐 vLLM 推理引擎。
> 旧 mrope 训练产出的 ckpt 在 vLLM 上接受率为 0%（transformers 5.x 会静默把 draft 的
> mrope 降级成 1D，导致训推不一致）。详见 memory `specforge-eagle3-vllm-aux-layer-mismatch`。

---

## 1. 环境

| 环境 | 用途 | 激活方式 |
|------|------|----------|
| `specforge` | 训练 + 数据导出 | `conda activate specforge` |
| `vllm` | 推理评测 (vLLM 0.21) | `conda activate vllm` |

Conda 根目录：`/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/minianaconda3`

---

## 2. 项目目录结构

```
SpecForge/                              # 项目根: /prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/SpecForge
│
├── model/                              # target 模型 (软链)
│   ├── Qwen2.5-VL-7B-Instruct → .../ViSpec/model/Qwen2.5-VL-7B-Instruct
│   ├── Qwen3.5-35B-A3B → .../haodong/hf_home/.../Qwen3.5-35B-A3B
│   ├── Qwen3.5-35B-A3B-DFlash/        # 35B DFlash draft (已训好)
│   └── ViSpec-Qwen2.5-VL-7B-Instruct → .../ViSpec/model/...
│
├── configs/                            # draft 模型 config 模板
│   ├── qwen2-5-vl-7b-eagle3.json      # ← 本任务用 (已改: 去掉 mrope, 加 aux 层)
│   ├── qwen2.5-vl-7b-dflash.json      # dflash VLM config
│   ├── qwen3.5-35b-a3b-dflash.json    # 35B dflash config
│   └── ...                             # 其他模型的 eagle3/dflash config
│
├── scripts/                            # 训练 & 工具脚本
│   ├── train_eagle3.py                 # ← eagle3 训练入口
│   ├── train_dflash.py                 # dflash 训练入口
│   ├── build_vocab_mapping_from_jsonl.py  # 构建 draft↔target 词表映射
│   ├── convert_vispec_to_eagle3.py     # ViSpec 数据→eagle3 格式
│   ├── convert_vispec_to_dflash.py     # ViSpec 数据→dflash 格式
│   ├── export_eval_datasets.py         # 导出评测集 (mmvet/sqa/mme/textvqa)
│   └── prepare_data.py / prepare_hidden_states.py  # 离线数据准备
│
├── specforge/                          # 训练框架核心
│   ├── core/
│   │   └── eagle3.py                   # ← eagle3 trainer (已改: position 3D→1D)
│   ├── modeling/
│   │   ├── draft/
│   │   │   └── llama3_eagle.py         # eagle3 draft model 定义 (LlamaForCausalLMEagle3)
│   │   └── target/
│   │       ├── eagle3_target_model.py  # target 侧 aux hidden 抽取
│   │       └── dflash_target_model.py  # dflash target 侧
│   ├── data/                           # 数据加载 (online_mm / eagle3_online_mm)
│   ├── distributed.py                  # FSDP / DP 分布式
│   └── utils.py                        # checkpoint 保存等
│
├── datasets/
│   ├── train/
│   │   └── qwen2.5-vl-7b-dflash.jsonl # 训练数据 (67919 条, eagle3/dflash 共用)
│   └── eval/                           # 评测集索引 + 图片 (由 export_eval_datasets.py 生成)
│       ├── mmvet_index.jsonl / mmvet_images/
│       ├── sqa_index.jsonl / sqa_images/
│       ├── mme_index.jsonl / mme_images/
│       └── textvqa_index.jsonl / textvqa_images/
│
├── cache/
│   └── vocab_mapping/
│       └── qwen2.5-vl-7b-eagle3.pt    # draft↔target 词表映射 (32000↔152064)
│
├── outputs/                            # 训练产出 checkpoints
│   ├── qwen2.5-vl-7b-eagle3/          # 旧版 (mrope, 接受率 0%, 废弃)
│   ├── qwen2.5-vl-7b-eagle3-1drope/   # ← 新版 (1D rope, 目标目录)
│   └── qwen2.5-vl-7b-dflash/          # dflash VLM ckpt (接受率 ~0.13, 1.29x)
│
├── results/                            # 评测结果 JSON + server log
├── examples/                           # 各模型的训练/评测 shell 脚本
│   ├── run_qwen2_5_vl_7b_dflash_eval.sh
│   ├── run_qwen2_5_vl_7b_eagle3_online.sh
│   └── ...
│
├── benchmark_sd_vlm.py                 # VLM 投机解码评测脚本 (vLLM/SGLang 双引擎)
├── benchmark_sd.py                     # 纯文本版评测 (MT-Bench)
├── docs/                               # 文档
│   ├── eagle3_vlm_training_guide.md    # ← 本文件
│   ├── dflash_architecture.md
│   └── dflash_model_spec.md
└── tests/                              # 测试
```

---

---

## 3. 训练数据

| 字段 | 说明 |
|------|------|
| 原始数据 | `datasets/train/gen_mm_combined/` (60000 条 .pt, online_0/1/2 三子目录) |
| 转换后 | `datasets/train/gen_mm_combined.jsonl` (需先跑转换脚本) |
| 格式 | 每行 JSON: `{"input_ids": [...], "loss_mask": [...], "image_file": "abs_path"}` |
| 图片基路径 | `/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/ViSpec` |
| chat_template | `qwen2-vl` |

### 数据转换 (首次需要)

`gen_mm_combined` 是 .pt 目录格式,训练脚本的 `OnlineMMDataset` 读 JSONL,需先转:

```bash
conda activate specforge
python scripts/convert_gen_mm_to_jsonl.py \
    --input-dir datasets/train/gen_mm_combined \
    --output datasets/train/gen_mm_combined.jsonl \
    --image-base /prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/ViSpec
```

约 1-2 分钟,产出 ~60000 行 jsonl。

Vocab mapping: `cache/vocab_mapping/qwen2.5-vl-7b-eagle3.pt`
- draft 词表 32,000 → target 词表 152,064 的双向映射 (d2t / t2d)
- 由 `scripts/build_vocab_mapping_from_jsonl.py` 生成

---

## 4. Draft 模型配置

文件：`configs/qwen2-5-vl-7b-eagle3.json`

```json
{
  "architectures": ["LlamaForCausalLMEagle3"],
  "model_type": "llama",
  "target_model_type": "qwen2_5_vl",
  "hidden_size": 3584,
  "intermediate_size": 18944,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "num_hidden_layers": 1,
  "head_dim": 128,
  "hidden_act": "silu",
  "max_position_embeddings": 8192,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000,
  "vocab_size": 152064,
  "draft_vocab_size": 32000,
  "eagle_aux_hidden_state_layer_ids": [2, 14, 25],
  "bos_token_id": 151643,
  "eos_token_id": 151645
}
```

**关键设计决策**：
- **无 `rope_scaling`** → draft 用 1D `LlamaRotaryEmbedding`（不用 mrope）
- **无 `eagle_aux_hidden_state_layer_ids`** → 不写此字段,让两个引擎各走默认公式。原因:vLLM 和 SGLang 对此字段的解读不兼容(vLLM 直接用值做含-embedding-offset 索引;SGLang 对值加 1 再用);但两者的默认公式 `[2, num//2, num-3]` 对 28 层 target 均产出等效的 decoder layer 1/13/24
- **`draft_vocab_size: 32000`** → eagle3 用精简词表 + d2t/t2d 映射

---

## 5. 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--target-model-path` | `./model/Qwen2.5-VL-7B-Instruct` | target VLM |
| `--draft-model-config` | `./configs/qwen2-5-vl-7b-eagle3.json` | draft config (1D rope) |
| `--train-data-path` | `./datasets/train/gen_mm_combined.jsonl` | 训练数据 (转换后) |
| `--vocab-mapping-path` | `./cache/vocab_mapping/qwen2.5-vl-7b-eagle3.pt` | 词表映射 |
| `--output-dir` | `./outputs/qwen2.5-vl-7b-eagle3-1drope` | 输出 |
| `--num-epochs` | 6 | 训练轮数 |
| `--batch-size` | 1 | 每卡 batch (online 模式只能 1) |
| `--learning-rate` | 1e-4 | 学习率 |
| `--max-length` | 4096 | 最大序列长度 |
| `--warmup-ratio` | 0.04 | 学习率 warmup 比例 |
| `--max-grad-norm` | 1.0 | 梯度裁剪 |
| `--ttt-length` | 7 | TTT 展开步数 (训练时模拟推理 7 步) |
| `--save-interval` | 7000 | 每 7000 步存一次 ckpt |
| `--log-interval` | 50 | 每 50 步 log |
| `--dp-size` | 8 | 数据并行卡数 |
| `--tp-size` | 1 | 张量并行 (eagle3 draft 小, 不需要) |
| `--attention-backend` | `flex_attention` | 注意力实现 |
| `--seed` | 0 | 随机种子 |
| `--is-vlm` | (flag) | 标记 target 为 VLM |
| `--vlm-online-mm` | (flag) | 在线处理图像 (不预抽特征) |
| `--chat-template` | `qwen2-vl` | 对话模板 |
| `--kl-scale` | 1.0 | KL loss 系数 |
| `--kl-decay` | 3.0 | KL 衰减 |
| `--target-model-backend` | `custom` | target 推理后端 |
| `--max-pixels` | 1003520 | VLM 图像最大像素 (≈1280×784) |
| `--min-pixels` | 200704 | VLM 图像最小像素 (≈512×392) |
| `--dataloader-num-workers` | 4 | 数据加载进程数 |
| `--report-to` | `tensorboard` | 训练日志 |
| `--resume` | (flag) | 断点续训 |

---

## 6. 训练命令

```bash
# 1. 激活 specforge 环境
source /prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/minianaconda3/etc/profile.d/conda.sh
conda activate specforge

# 2. 进入项目目录
cd /prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/SpecForge

# 3. 8 卡数据并行训练
torchrun --nproc_per_node=8 scripts/train_eagle3.py \
    --target-model-path ./model/Qwen2.5-VL-7B-Instruct \
    --draft-model-config ./configs/qwen2-5-vl-7b-eagle3.json \
    --train-data-path ./datasets/train/gen_mm_combined.jsonl \
    --vocab-mapping-path ./cache/vocab_mapping/qwen2.5-vl-7b-eagle3.pt \
    --output-dir ./outputs/qwen2.5-vl-7b-eagle3-1drope \
    --is-vlm --vlm-online-mm \
    --chat-template qwen2-vl \
    --target-model-backend custom \
    --num-epochs 6 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --ttt-length 7 \
    --kl-scale 1.0 --kl-decay 3.0 \
    --dp-size 8 --tp-size 1 \
    --attention-backend flex_attention \
    --save-interval 7000 --log-interval 50 \
    --max-pixels 1003520 --min-pixels 200704 \
    --dataloader-num-workers 4 \
    --seed 0 \
    --report-to tensorboard
```

**断点续训**（如果训练中断）：加 `--resume --ckpt-dir ./outputs/qwen2.5-vl-7b-eagle3-1drope/epoch_X_step_Y`

**预估时间**：上次 mrope 版跑了 ~10 小时 (8×A100-80G, 67919 条×6 epochs = 50940 步)。

---

## 7. 训练产出

输出目录结构：
```
outputs/qwen2.5-vl-7b-eagle3-1drope/
├── epoch_0_step_7000/
│   ├── config.json          # draft config (自动从模型写入, 无 rope_scaling)
│   ├── model.safetensors    # draft 权重 (~800MB)
│   └── training_state.pt    # optimizer/scheduler/epoch/step
├── epoch_1_step_14000/
├── ...
├── epoch_5_step_50940/      # 最终 ckpt
└── runs/                    # TensorBoard 日志
    └── events.out.tfevents.*
```

**查看训练曲线**：
```bash
tensorboard --logdir outputs/qwen2.5-vl-7b-eagle3-1drope/runs --port 6006
```

关注指标：
- `train/acceptance_rate_*`: 采样解码期望接受率（应逐步上升到 0.7+）
- `train/acc_*`: 严格 argmax 匹配率（这个数值较低是正常的，不是 bug）
- `train/ploss_*`: 预测 loss（应从 ~6.5 降到 ~1.5）
- `train/grad_norm`: 梯度范数（收敛后应 <2）

---

## 8. 评测命令 (训练完成后)

新 ckpt 天然兼容 vLLM（1D rope + aux 层字段已在 config 里），直接评测：

```bash
# 切到 vllm 环境
conda activate vllm

# 单卡单请求 (测单请求加速比, 最标准)
CUDA_VISIBLE_DEVICES=0 python benchmark_sd_vlm.py \
    --engine vllm \
    --model-path ./model/Qwen2.5-VL-7B-Instruct \
    --draft-model-path ./outputs/qwen2.5-vl-7b-eagle3-1drope/epoch_5_step_50940 \
    --speculative-method eagle3 \
    --datasets mmvet,sqa,mme,textvqa \
    --num-questions 50 \
    --num-speculative-tokens 5 \
    --max-tokens 512 \
    --temperature 0.0

# 8 卡 DP + 高并发 (测系统吞吐)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python benchmark_sd_vlm.py \
    --engine vllm \
    --model-path ./model/Qwen2.5-VL-7B-Instruct \
    --draft-model-path ./outputs/qwen2.5-vl-7b-eagle3-1drope/epoch_5_step_50940 \
    --speculative-method eagle3 \
    --datasets mmvet,sqa,mme,textvqa \
    --num-questions 50 \
    --data-parallel-size 8 --concurrency 64
```

---

## 9. 本次改动 vs 旧版的差异

| | 旧版 (mrope, 接受率 0%) | 新版 (1D rope) |
|---|---|---|
| draft config | 有 `rope_scaling: {type:mrope, mrope_section:[16,24,24]}` | **无 `rope_scaling`** |
| 训练时 position | 3D mrope `[3, batch, seq]` | **1D** `[batch, seq]` (从 3D 取 `[0]`) |
| vLLM 推理兼容性 | transformers 5.x 把 mrope 降级成 1D → 训推不一致 → 0% | 训推都是 1D → 一致 |
| aux 层 | 未写入 config → vLLM 用默认 (2,14,25) 与训练 (1,13,24) 差 1 | **写入 config** → vLLM 自动用 (1,13,24) |

**代码改动清单**：
1. `configs/qwen2-5-vl-7b-eagle3.json` — 去掉 `rope_scaling`；加 `eagle_aux_hidden_state_layer_ids`
2. `specforge/core/eagle3.py` ~648 行 — `get_rope_index` 返回 3D 后加 `if position_ids.ndim == 3: position_ids = position_ids[0]`

---

## 10. 已知坑与排查指南

| 问题 | 现象 | 解法 |
|------|------|------|
| vLLM 8卡DP 死锁 | GPU 满显存但 util=0, health=000, EngineCore 卡在 `pipe_write` | 脚本已修: server 输出落日志文件 (非 PIPE) |
| TP=8 不可用 | vLLM 启动报 `num_heads % tp != 0` | Qwen2.5-VL-7B 28 头, 只能 TP=1/2/4, 用 DP=8 代替 |
| aux 层不匹配 | 接受率 ~0 (即使 1D rope 对了) | 确保 config 有 `eagle_aux_hidden_state_layer_ids: [2,14,25]`（含 embedding 的索引） |
| HOME 配额写爆 | torch.compile 卡住报 Disk quota exceeded | `export VLLM_CACHE_ROOT=/china-scratch/.../cache` |
| 训练 acc vs acceptance_rate 差 2000 倍 | 正常,不是 bug | `acc`=严格 argmax 匹配(对应 T=0); `acceptance_rate`=采样理论值(对应 T>0) |
