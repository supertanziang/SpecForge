# DFlash VLM 离线数据转换：`convert_vispec_to_dflash.py`

本文讲清楚 DFlash 图文（VLM）训练这条线上**最上游、也最关键的一环**：如何把
ViSpec 预生成的多模态 `.pt` 数据，转换成 DFlash 训练直接消费的 JSONL。

> 阅读前置：训练侧如何加载这份 JSONL、如何在线算 target hidden，见
> `specforge/data/online_mm_dataset.py` 与
> `specforge/modeling/target/dflash_target_model.py`（`HFDFlashTargetModel`）。
> 本文只覆盖「离线生成」这一步。

---

## 1. 这一环在整个 pipeline 的位置

完整的 VLM DFlash 训练数据链路：

```
ViSpec 生成流程
  (目标模型 Qwen2.5-VL 对图文 prompt 做真实长文本生成)
        │
        ▼
  ViSpec .pt 文件          ← full_ids / prompt_len / image_file
        │
        │  ★ convert_vispec_to_dflash.py（本文）
        ▼
  DFlash 训练 JSONL        ← input_ids / loss_mask / image_file
        │
        │  OnlineMMDataset + prepare_mm_dataloader（bs=1）
        ▼
  训练循环 generate_dflash_data
  (hf Qwen2_5_VLForConditionalGeneration 在线前向，取多层 hidden)
        │
        ▼
  DFlash draft model 训练
```

要点：**token 序列在 ViSpec 阶段就已经定型了**（包括图像占位 token 的展开），
转换脚本不做任何 tokenize、不套对话模板、不跑视觉塔。它只是一次**纯格式搬运 +
loss_mask 构造 + 过滤**。这也是为什么训练脚本 `run_qwen2_5_vl_7b_dflash_online.sh`
里没有 `--chat-template`：模板早在 ViSpec 生成 `full_ids` 时就套好了。

---

## 2. 输入：ViSpec `.pt` 格式

每个 `.pt` 文件是一条样本，`torch.load` 后是一个 dict：

| 字段 | 类型 | 含义 |
|---|---|---|
| `full_ids` | `Tensor[S]` long（也可能是 `[1, S]`） | 完整 token 序列 = prompt + 目标模型生成的长回复。**已包含展开后的图像占位 token**（Qwen2.5-VL `image_token_id=151655`） |
| `prompt_len` | int | prompt 段长度；生成段 = `full_ids[prompt_len:]` |
| `image_file` | str | 相对 ViSpec 根目录的图片路径 |

关键理解 —— **图像占位 token 已经 baked 进 `full_ids`**：

Qwen2.5-VL 的输入里，一张图会被展开成一长串 `<|image_pad|>`（id=151655）占位
token，数量由图像分辨率经 patch / merge 换算而来。ViSpec 在生成时用了某个固定的
`min_pixels / max_pixels` 配置，按那个分辨率把占位 token 数算死并写进了
`full_ids`。**下游训练时必须用完全相同的 `min/max-pixels`**，否则 processor 重新
算出的 `image_grid_thw` 与序列里 baked 的占位 token 数对不上，前向直接 crash
（见 `train_dflash.py` 中 `--min-pixels/--max-pixels` 的告警注释）。

---

## 3. 输出：DFlash 训练 JSONL 格式

每行一个 JSON 对象：

| 字段 | 类型 | 由什么得到 |
|---|---|---|
| `input_ids` | `list[int]` | `= full_ids`（截断到 `--max-len`） |
| `loss_mask` | `list[int]` | `= [0]*prompt_len + [1]*gen_len`，长度 == `len(input_ids)` |
| `image_file` | str | 绝对路径（`--image-root` 拼接 ViSpec 相对路径），训练时 `Image.open` 用 |

**`attention_mask` 故意不落盘**：它在 bs=1 下恒为全 1，训练时按 `input_ids` 长度
现生成即可（见 `OnlineMMCollator`），落盘只是浪费体积。

**`loss_mask` 的语义**：`0` = prompt 段（含全部图像占位 token，不计 loss），
`1` = 生成段（draft model 要学着预测的部分）。这与纯文本 EAGLE3/DFlash 的 loss_mask
语义一致，只是这里 prompt 段天然把图像 token 一并屏蔽掉了。

---

## 4. 转换逻辑逐步拆解

### 4.1 递归收集 `.pt`（`list_pt_files`）

```python
for root, _dirs, fnames in os.walk(src, followlinks=True):
    ...  # 收集所有 *.pt
```

注意 `followlinks=True`：`gen_mm_combined` 目录下往往是用**软链接**把多份数据集
拼装在一起的，必须跟随软链接才能扫到真正的文件。结果按文件名排序，保证可复现。

### 4.2 读取与截断

```python
rec = torch.load(fp, map_location="cpu", weights_only=False)
full_ids = rec["full_ids"]
if full_ids.dim() == 2:        # 兼容 [1, S] 形态
    full_ids = full_ids[0]
full_ids = full_ids[: args.max_len]   # 截断，与训练 --max-length 对齐
seq = full_ids.shape[0]
prompt_len = min(int(rec["prompt_len"]), seq)  # 截断后 prompt_len 不能超过 seq
```

`--max-len` 必须与训练脚本的 `--max-length`（4096）保持一致——否则截断长度不同，
样本会和训练时的实际可见长度产生偏差。

### 4.3 构造 `loss_mask` 并做长度过滤

```python
gen_len = seq - prompt_len
if gen_len < args.min_gen_tokens:   # 生成段太短的样本丢弃
    n_skip_short += 1
    continue
loss_mask = [0] * prompt_len + [1] * gen_len
```

`--min-gen-tokens`（默认 32）是为了对齐 DFlash 训练侧的过滤逻辑（纯文本路径里是
`loss_mask.sum() >= 2*block_size`，block_size=16 即 32）。生成段太短的样本对 draft
训练几乎没有监督信号，提前丢掉省得训练时再过滤。

### 4.4 图片路径拼接与（可选）存在性校验

```python
abs_image = image_file if os.path.isabs(image_file) \
            else os.path.join(args.image_root, image_file)
if args.check_image and not os.path.exists(abs_image):
    n_skip_noimg += 1
    continue
```

ViSpec 里 `image_file` 是相对路径，这里用 `--image-root` 拼成绝对路径写进
JSONL，训练时 `HFDFlashTargetModel` 直接 `Image.open(image_file)`。`--check-image`
默认关闭（逐文件 stat 很慢）；首次转换一批新数据时建议开一次，确认图片都在。

### 4.5 写出

```python
fout.write(json.dumps({
    "input_ids": full_ids.tolist(),
    "loss_mask": loss_mask,
    "image_file": abs_image,
}, ensure_ascii=False) + "\n")
```

`ensure_ascii=False` 保留路径里的非 ASCII 字符原样。

---

## 5. 用法

```bash
python scripts/convert_vispec_to_dflash.py \
    --src        datasets/train/gen_mm_combined \
    --out        datasets/train/qwen2.5-vl-7b-dflash.jsonl \
    --image-root /prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/ViSpec \
    --max-len    4096 \
    --min-gen-tokens 32 \
    --check-image          # 可选，首次转换时建议开
```

运行结束会打印：写出多少条、因生成段过短跳过多少条、（开了 `--check-image` 时）
因图片缺失跳过多少条。

---

## 6. 与下游训练绑死的约束（最容易踩坑的地方）

| 约束 | 原因 | 不一致的后果 |
|---|---|---|
| 转换时的 `min/max-pixels` 必须 == 训练 `--min-pixels/--max-pixels` 必须 == ViSpec 生成时的配置 | 图像占位 token 数在 `full_ids` 里 baked 死了 | processor 重算 `image_grid_thw` 对不上占位 token 数，前向 **crash** |
| `--max-len` == 训练 `--max-length` | 截断点要一致 | 样本可见长度与训练不符 |
| `--min-gen-tokens` ≈ `2 * block_size` | 对齐 DFlash 监督信号下限 | 留下无效样本或过滤标准不一 |
| `--image-root` 拼出的路径在训练机上可访问 | 训练时 `Image.open` 直接读绝对路径 | 训练 step 报 FileNotFound |
| 下游 bs 固定为 1 | 每张图 `pixel_values` 形状随分辨率变化，batch 内难拼接 | `OnlineMMCollator` 直接 `assert len==1` |

---

## 7. 训练侧差异：VLM-7B vs 纯文本 35B-A3B

数据进入训练后，VLM 这条线（`run_qwen2_5_vl_7b_dflash_online.sh`）与纯文本
大模型线（`run_qwen3.5_35b_a3b_dflash_online.sh`）还有一组结构性区别。先给总判断：

> **训练算法与主循环完全共享**（同一段 `train_dflash.py` 的训练 loop）：loss 计算、
> `backward`、`BF16Optimizer`、FSDP 包装、checkpoint 保存、学习率调度，两条线一模一样。
> 真正的区别集中在 **draft model 结构、从 target 哪里取 hidden、词表/mask 接入、
> 以及 target 模型怎么进显存**这四块。

### 7.1 draft model 结构（来自各自的 `configs/*-dflash.json`）

draft model 是 `DFlashDraftModel(draft_config)`，结构由 config 决定：

| 字段 | 35B-A3B draft | VLM-7B draft | 说明 |
|---|---|---|---|
| `hidden_size` | 2048 | 3584 | **必须 == target 的 hidden_size**（硬约束，不是超参） |
| `num_hidden_layers`(draft) | 8 | 5 | draft 自身层数 |
| `num_attention_heads` | 32 | 28 | 跟随各自 target |
| `num_key_value_heads` | 4 | 4 | 相同 |
| `intermediate_size` | 6144 | 18944 | VLM 的 FFN 宽得多 |
| `num_target_layers` | 40 | 28 | target 总层数 |
| `target_layer_ids` | [1,10,19,28,37] | [1,7,13,19,25] | 抓 target 哪几层 hidden 拼接 |
| `vocab_size` | 248320 | 152064 | 词表不同 |
| `rope_theta` | 1e7 | 1e6 | |
| `target_model_type` | （无） | `qwen2_5_vl` | VLM config 显式标注 |

**最关键的是 `hidden_size` 必须等于 target**：draft 直接吃 target 的 hidden state，
维度不对齐会立刻报错。

### 7.2 target hidden 从哪几层取（`target_layer_ids`）

DFlash 不是只用 target 最后一层，而是**抓中间 5 层拼接**喂给 draft
（`build_models` 里 `target_model.set_capture_layers(draft_model.target_layer_ids)`）。
两边都取 5 层、拼成 `hidden_size * 5` 维，但层 id 与提取机制不同：

- **35B（sglang）**：靠 `set_eagle3_layers_to_capture` 注入引擎，从
  `output.aux_hidden_states` 取（`dflash_target_model.py:115-119, 169-174`）。
- **VLM（hf）**：靠 `output_hidden_states=True`，从 `outputs.hidden_states[idx+1]`
  手动选层 concat（`dflash_target_model.py:312-318`，`+1` 跳过 embedding 层）。

> 训练与推理的 `target_layer_ids` 必须一致，否则会出现接受率为 0 的现象。

### 7.3 词表 / mask_token / embed-head 接入

| 项 | 35B | VLM |
|---|---|---|
| `vocab_size` | 248320 | 152064 |
| `mask_token_id` | 248070（自带空闲槽/自动添加） | 151657（**强制显式 `--mask-token-id`**） |
| embed/head key | `model.language_model.embed_tokens.weight`（MoE 多一层命名） | 默认 `model.embed_tokens.weight` |
| tokenizer 来源 | `AutoTokenizer` | `AutoProcessor.tokenizer` |

`train_dflash.py:468-476` 有硬性分支：**VLM 必须显式给 `--mask-token-id`**——VLM 的
embedding 冻结、不 resize，不能像文本那样自动 `add_special_tokens` 新增 `<|MASK|>`
（会越界），必须指定一个空闲槽位（Qwen2.5-VL 用 151657）。`TargetEmbeddingsAndHead`
加载的 target embed + lm_head 两边都**冻结复用、不训**，只是 key 路径不同。

### 7.4 target 模型怎么进显存

| 项 | 35B（sglang 后端） | VLM（hf 后端） |
|---|---|---|
| 加载方式 | 每个 rank 起 `SGLangRunner` 推理引擎 | `Qwen2_5_VLForConditionalGeneration.eval().to("cuda")` |
| 显存 | `--sglang-mem-fraction-static 0.5` 划走半张卡 | 整模型直接上 GPU |
| 前向 | `Req → ScheduleBatch → ForwardBatch` 引擎调度，`disable_cuda_graph` | 普通 `model(**inputs)`，跑视觉塔注入图像特征 |
| 模型类 | sglang 内部加载 | **ConditionalGeneration**（非 `AutoModelForCausalLM`），否则视觉塔不跑 |

`get_dflash_target_model`（`dflash_target_model.py:338-343`）有一道硬拦截：
**sglang + is_vlm 直接报错**——sglang backend 不返回图像感知 hidden。所以后端选择
不是自由的，而是被 VLM 特性强制的（VLM 只能 hf，大 MoE 才用 sglang）。

### 7.5 batch / 数据并行 / resume

| 项 | 35B | VLM |
|---|---|---|
| batch | bs=2，`prepare_dp_dataloaders` 多样本 DP 分片 | bs=1 强制，`prepare_mm_dataloader` |
| 样本过滤 | 训练时 `loss_mask.sum() >= 2*block_size` | 离线转换时已按 `--min-gen-tokens` 过滤 |
| resume | `--resume` 断点续训 + skip_steps 跳过已训 step | 脚本未开 |

dflash 算法核心超参（`num-anchors=512`、`loss-decay-gamma=7.0`、`block-size=16`、
`lr=6e-4`、`warmup-ratio=0.04`、`max-grad-norm=1.0`、`max-length=4096`）两条线**完全一致**
——印证了「算法不变，变的只是 target 形态」这一结论。

---

## 8. 一句话总结

> `convert_vispec_to_dflash.py` 是一次**纯搬运 + loss_mask 构造 + 过滤**：把 ViSpec
> 已经定型好（含展开图像占位 token）的 `full_ids` 原样转成 `input_ids`，按
> `prompt_len` 切出 `loss_mask`，把图片相对路径拼成绝对路径。它不 tokenize、不套
> 模板、不跑视觉塔——所有「智能」都在上游 ViSpec 生成阶段完成了。唯一要时刻警惕的
> 是 **`min/max-pixels` 三处一致**（ViSpec 生成 / 本转换隐含 / 训练加载），否则图像
> token 数对不上，训练前向必崩。
