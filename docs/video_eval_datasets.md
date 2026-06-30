# 视频理解评测集构造说明（Video-MME / LongVideoBench）

> 适用于 SpecForge dflash VLM 投机解码加速比评测。
> 涉及脚本：`scripts/export_video_eval_datasets.py`（导出）、`benchmark_sd_vlm.py`（评测）。

---

## 一、整体设计哲学

视频评测集走的是 **「视频文件 + jsonl 索引」分离** 的路线，刻意 **不** 用 `datasets.load_dataset`。核心两个脚本：

- `scripts/export_video_eval_datasets.py`（在 `specforge` 环境跑）：下载视频 + 生成索引
- `benchmark_sd_vlm.py`（在 `vllm` 环境跑）：抽帧 + 发请求测加速比

之所以切开，原因写在脚本注释里：`datasets.load_dataset` / `list_repo_files` 会拉取 **整个仓库文件树**（Video-MME 视频仓库有 80w+ 文件），极易触发 HF 429 限流。所以改成：

- 只下载 **单个元数据文件**（parquet / `lvb_val.json`）
- 用 `HfFileSystem.ls` **非递归** 列视频目录
- 用 `hf_hub_download` **单文件** 下载视频

---

## 二、两个数据集的来源

| 数据集 | 元数据来源 | 视频来源 | 任务类型 |
|---|---|---|---|
| **Video-MME** | `lmms-lab/Video-MME` 的 parquet | `vid-modeling/videomme` 的 `data/` 下约 155 个 YouTube 视频 | 视频理解多选问答 |
| **LongVideoBench** | `lccshunli/LongVideoBench` 的 `lvb_val.json` | 同仓库 `videos/videos/<id>.mp4` | 长视频推理问答 |

> **关键细节**：Video-MME 的视频仓库只有约 155 个 YouTube 视频可用（其余受版权限制下不到），所以代码用 `HfFileSystem.ls` 非递归列出 `data/` 下实际存在的 mp4，再和 parquet 元数据 **取交集**，每个视频只取第一题。

---

## 三、下载到哪里了

两层结构：**索引在仓库内，视频在 HF 缓存里（仓库内目录是软链接）**。

### 索引文件（仓库内，纯文本）

```
datasets/eval/videomme_index.jsonl        # 50 条
datasets/eval/longvideobench_index.jsonl  # 50 条
```

### 视频目录（仓库内，全是软链接）

```
datasets/eval/videomme_videos/        # 50 个 .mp4 软链
datasets/eval/longvideobench_videos/  # 49 个 .mp4 软链
```

### 软链接真正指向的位置（HF 缓存，真实文件在这）

统一在 HF cache 根 `…/ziantan/.cache/huggingface/hub/` 下：

| 数据集 | 真实文件路径 |
|---|---|
| **Video-MME** | `…/.cache/huggingface/hub/datasets--vid-modeling--videomme/snapshots/bcfd98d…/data/<id>.mp4` |
| **LongVideoBench** | `…/.cache/huggingface/hub/datasets--lccshunli--LongVideoBench/snapshots/a8be12c…/videos/videos/<id>.mp4` |

这正是导出脚本里 `os.symlink(src, local_path)` 干的事——`hf_hub_download` 把文件下到 HF 缓存返回 `src`，再在仓库 `datasets/eval/<name>_videos/` 下建软链 `local_path`，所以仓库目录 **不占额外磁盘**。

> **注意**：videomme 索引 50 条但 lvb 视频目录只有 49 个文件——因为 lvb 同一个视频可能出多道题（id 形如 `<vid>_0`），多条索引共享同一个 mp4 软链。

---

## 四、索引格式（统一 schema）

每个数据集导出成 `datasets/eval/<name>_index.jsonl`，每行一条：

```json
{
  "id": "539-1",
  "question": "How many new McDonald's... A. 95. B. 221. ...\nFirst describe what you observe...",
  "video_path": "/.../videomme_videos/FcP0mzWFCQU.mp4",
  "duration": 741.7,
  "num_frames": 16,
  "dataset": "videomme"
}
```

| 字段 | 含义 |
|---|---|
| `id` | 题目唯一标识 |
| `question` | 题干 + 选项 + 强制拼接的 `ANSWER_SUFFIX` |
| `video_path` | 仓库内软链接路径（指向 HF 缓存的真实 mp4） |
| `duration` | 视频时长（秒） |
| `num_frames` | 抽帧数（按时长自适应，见下） |
| `dataset` | `videomme` / `longvideobench` |

两个集各 **50 条**。

---

## 五、两个最关键的构造决策

### 1. `ANSWER_SUFFIX`：诱导长输出

多选题原本只输出一个选项字母（约 2 token），decode 阶段太短，**无法体现投机解码的加速**（SD 加速的正是 decode）。所以每道题后面强行拼一段：

> "First describe what you observe in the video relevant to the question, then reason step by step, and finally answer with the option letter."

逼模型先描述视频、再逐步推理、最后给字母，产生中长输出，这样 speedup / accept_rate 才有统计意义。这是对齐图文版 textvqa 的做法。

### 2. `adaptive_num_frames`：按时长自适应帧数

```
≤10s   → 4 帧
10-30s → 8 帧
>30s   → 16 帧（上限）
```

16 帧是硬上限，用来 **控 token 预算**——注释明说「32 帧 × 长视频会超过 8k 上下文」。

实际分布：

- **videomme**：50 条，时长 29.5s ~ 3472s（中位 396.8s），帧数几乎全是 16（49 条 16 帧，1 条 8 帧）
- **longvideobench**：50 条，时长 9s ~ 3135s（中位 918.4s），44 条 16 帧、5 条 8 帧、1 条 4 帧

> ⚠️ **重要坑**：视频普遍很长但帧数被钉死在 16，**16 帧的长 prefill 主导了延迟**，导致 7B/35B 的 speedup 都 <1（0.83-0.90x），接受率即便不低也转不成加速。

---

## 六、评测端如何消费（抽帧落在这里）

注意：**导出阶段只下载 mp4 + 算时长写 `num_frames`，并不抽帧**。真正抽帧在 `benchmark_sd_vlm.py` 的 `extract_video_frames`，发请求时才做：

```python
# 均匀抽帧：第 i 帧取 total_frames * i / num_frames
indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
```

**均匀采样**：把整段视频按帧数等分，取每段的**起点帧**。分母是 `num_frames`（不是 `num_frames-1`），所以第 0 帧一定是开头（idx=0），但**最后一帧取不到结尾**（最后索引 ≈ 93.75% 处）。无关键帧检测、无 resize、读帧失败静默跳过。

每个 index 走：

```python
cap.set(cv2.CAP_PROP_POS_FRAMES, idx)               # 定位到该帧
ret, frame = cap.read()                             # 读出 BGR ndarray
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR→RGB
img = Image.fromarray(frame_rgb)
img.save(buf, format="JPEG", quality=85)            # 编码 JPEG q=85
b64 = base64.b64encode(...)                         # → data:image/jpeg;base64,...
```

最终 N 帧变成 **N 个 `image_url`**，按时间顺序 + 末尾一个 text question 一起塞进 chat content。

**实跑示例**（例 1 的 `FcP0mzWFCQU.mp4`，total_frames=17784, fps=23.98, num_frames=16）：

| 第 i 帧 | 帧索引 | 时间点 |
|---|---|---|
| 0 | 0 | 0.0s |
| 1 | 1111 | 46.3s |
| … | … | …（每隔 ≈46.3s 一帧） |
| 14 | 15561 | 649.0s |
| 15 | 16672 | 695.4s |

整段 741.7s，**最后 46s（695s→741s）完全没采到**——这是 `total/num_frames` 而非 `total/(num_frames-1)` 的固有偏差，对「结尾才出答案」的题目不利。

> 这一点很重要——评测端其实是把视频「降级」成一组图片喂模型的，并不走 vLLM 原生的 video 输入。
> opencv 对非关键帧的 seek 在某些编码下会落到最近关键帧，所以「时间点」是名义值，实际解出帧可能有几十毫秒偏移。

---

## 七、具体例子

### 例 1：Video-MME（`id: 539-1`）

- **video_path**：`datasets/eval/videomme_videos/FcP0mzWFCQU.mp4`（→ HF 缓存，**48 MB**）
- **duration**：741.7s（约 12 分钟）→ 落入 `>30s` 档 → **num_frames: 16**
- **question**（注意末尾被强制拼了 `ANSWER_SUFFIX`）：

  > How many new McDonald's have opened in France from 2018-2023 in the video?
  > A. 95. B. 221. C. 83. D. 92.
  > *First describe what you observe in the video relevant to the question, then reason step by step, and finally answer with the option letter.*

  选项是 4 个（`A. B. C. D.`），格式来自 parquet 的 `options` 字段直接 `" ".join`。

### 例 2：LongVideoBench（`id: 6hBbXVkgxGE_0`）

- **video_path**：`datasets/eval/longvideobench_videos/6hBbXVkgxGE.mp4`（→ HF 缓存，**41 MB**）
- **duration**：539.6s（约 9 分钟）→ **num_frames: 16**
- **question**：

  > A person dressed in black clothes is making a phone call. One of his hands is placed on the table. On the table, there is a desk lamp. Next to the desk lamp is an open window. To the left of the window is a cabinet, on which there are many bottles of alcohol. Who is making the phone call?
  > (A) Jan (B) Dougie (C) Bullet (D) Mike (E) Statham
  > *First describe what you observe in the video relevant to the question, then reason step by step, and finally answer with the option letter.*

  选项 **5 个**（`(A)…(E)`），格式来自 `lvb_val.json` 的 `candidates` 列表，用 `f"({chr(65+i)}) {c}"` 现拼。

---

## 八、两个集的样本对比（构造差异）

| | Video-MME | LongVideoBench |
|---|---|---|
| 选项数 | 4（A./B./C./D.） | 不定，常 5（(A)/(B)/…） |
| 选项来源 | parquet `options` 字段 | json `candidates` 列表，代码现拼字母 |
| 问题风格 | 直接问事实（数字/事件） | 先给一大段视觉场景描述再提问 |
| id 格式 | `<question_id>`（如 `539-1`） | `<video_id>_<序号>`（如 `6hBbXVkgxGE_0`） |
| 打乱方式 | `random.Random(42).shuffle` 固定种子 | 同左 |

两个集都经固定种子打乱后取前 50，保证可复现。

---

## 九、评测结果（dflash 投机解码）

> 数据来自 `results/` 下两个有效结果文件（n=50 且 baseline 完整）：
> - `Qwen2.5-VL-7B-Instruct-vllm-dflash_..._063910.json`
> - `Qwen3.5-35B-A3B-vllm-dflash_..._075648.json`
>
> 其余冒烟测试（n=3）和缺 baseline 的残缺文件已删除。

### 设置

两个模型共享：`method=dflash`、`k(num_speculative_tokens)=5`、`temperature=0.0`（贪心，固定 seed）、`enable_thinking=true`、`concurrency=1`、`num_questions=50/集`、engine=vLLM 0.21.0。

不同点：

| | Qwen2.5-VL-7B | Qwen3.5-35B-A3B |
|---|---|---|
| tensor_parallel_size | **1** | **2** |
| max_tokens | **512** | **4096** |
| draft model | Qwen2.5-VL-7B dflash | Qwen3.5-35B-A3B-DFlash |

### 结果

> json 里 `accept_length` 字段为 `null`，下表由 `accepted_tokens / draft_tokens / k` 反推：步数 = `draft_tokens/k`，accept_length = `(accepted_tokens + 步数)/步数`（即每次 target 前向平均吐出的 token 数，1.0 = 完全未接受）。

| 模型 | 数据集 | accept_rate | **accept_length** | SD tps | baseline tps | **speedup** |
|---|---|---|---|---|---|---|
| **7B** (TP=1) | videomme | 0.149 | **1.75** | 18.4 | 20.6 | **0.896** |
| | longvideobench | 0.108 | **1.54** | 18.2 | 20.4 | **0.895** |
| **35B** (TP=2) | videomme | 0.409 | **3.05** | 120.8 | 135.0 | **0.895** |
| | longvideobench | 0.394 | **2.97** | 119.4 | 143.3 | **0.833** |

### 结论

- **全线 speedup < 1（0.83~0.90x），投机解码在视频上反而变慢**。
- **35B 的 accept_length ≈ 3.0、accept_rate ≈ 0.40，远强于 7B（1.5~1.75 / 0.11~0.15）**，draft 质量本身没问题；但接受长度优势**没能转成墙钟加速**。
- 根因：**16 帧的长 prefill 主导端到端延迟**——每帧上百 vision token，prefill 一次性几千 token，decode 阶段再快也被吃掉。对比图文场景 dflash 能到 1.31~1.38x，视频直接掉到 <1。
- ⚠️ **TP 不同（7B=1 / 35B=2），speedup 不能跨模型直接比**，两个 0.89 不是一回事。

---

## 十、prefill / decode 时间占比实测（定位瓶颈）

> 上一节只给出端到端 speedup<1 的结论，本节用离线 profiling 把端到端延迟拆成
> **TM（target 35B）/ DM（draft）× prefill / decode** 共 4 段，定位时间到底卡在哪。
>
> **被测模型**：Qwen3.5-35B-A3B（TP=2，bf16，vLLM 0.21.0）+ Qwen3.5-35B-A3B-DFlash draft（k=5）。
> **样本**：每个数据集取索引第一条（与第七节「具体例子」同一条）。
> **帧数**：用第五节放开 16 帧上限后的新自适应帧数（videomme 25 帧 / lvb 18 帧）。
> **测量手段**：
>   - TM 用 vLLM 流式接口，`TTFT` = prefill（vision encode + LLM prefill），`(total-TTFT)/(tok-1)` = 每 token decode；脚本 `scripts/profile_tm_video.py`。
>   - DM 用 HF 单独加载（仅 908MB、8 层），`cuda.Event` 分别计 context K/V 投影（prefill）与一个 draft step（decode）；脚本 `scripts/profile_dflash_dm.py`。

### 10.1 四段原始实测

| 指标 | Video-MME（25 帧） | LongVideoBench（18 帧） |
|---|---|---|
| prompt_tokens（含 vision token） | 22163 | 16022 |
| accept_length（实测反推） | 3.05 | 2.97 |
| **TM-prefill (TTFT)** | **2574 ms** | **1314 ms** |
| **TM-decode / token** | **5.86 ms** | **5.76 ms** |
| **DM-prefill**（context K/V 投影，一次） | 9.37 ms | 6.71 ms |
| **DM-draft / step**（6 query token 一次 forward） | 16.55 ms | 12.89 ms |

### 10.2 完整时间占比分解（以 Video-MME 样本为例，生成 512 token）

> 样本：`id=539-1`，`FcP0mzWFCQU.mp4`，741.7s，25 帧，prompt_tokens=22163，accept_length=3.05。

**① baseline（无投机，纯 TM 逐 token decode）= 5574 ms**

| 阶段 | 耗时 | 占比 |
|---|---|---|
| TM-prefill | 2574 ms | **46.2%** |
| TM-decode（512 × 5.86ms） | 3000 ms | 53.8% |

**② SD（vLLM cache 化，167 轮 × accept_len 3.05）= 6345 ms**

> 计算口径：轮数 = 512 / 3.05 ≈ 167。SD 总时间 = TM-prefill + DM-prefill（context KV 只算一次）+ 167×(TM-verify + DM-draft)。

| 阶段 | 耗时 | 占比 | 说明 |
|---|---|---|---|
| TM-prefill | 2574 ms | 40.6% | ⛔ 投机解码无法加速 |
| DM-prefill | 9 ms | 0.1% | context KV 只算一次 |
| TM-verify（167 轮） | 984 ms | 15.5% | 每轮一次 target 前向 |
| DM-draft（167 轮） | 2778 ms | **43.8%** | ⚠️ cross-attention 吃满 22k context |

### 10.3 单轮 vs 整样本：TM-verify 与 DM-draft 的 decode 占比

一次投机解码迭代（一「轮」）= **DM-draft（draft 出 k=5 个候选 token，一次 forward）** + **TM-verify（target 对候选做一次并行验证）**。每轮平均接受 `accept_length` 个 token，所以生成 512 token 需要 `总轮数 = 512 / accept_length` 轮。

#### Video-MME 样本（25 帧，accept_length=3.05，总轮数 ≈ 168）

**单轮 decode** = 22.41 ms

| 阶段 | 单轮耗时 | 占比 |
|---|---|---|
| **DM-draft**（draft k=5 token） | 16.55 ms | **73.9%** |
| **TM-verify**（target 验证） | 5.86 ms | 26.1% |

**整样本 decode 段**（168 轮 + 1 次 DM-prefill）= 3771 ms

| 阶段 | 累计耗时 | 占比 |
|---|---|---|
| DM-prefill（context KV，一次） | 9.4 ms | 0.2% |
| **DM-draft 累计（168 轮）** | 2778 ms | **73.7%** |
| **TM-verify 累计（168 轮）** | 984 ms | 26.1% |

> DM 合计（prefill+draft）73.9% vs TM-verify 26.1%。

#### LongVideoBench 样本（18 帧，accept_length=2.97，总轮数 ≈ 172）

**单轮 decode** = 18.65 ms

| 阶段 | 单轮耗时 | 占比 |
|---|---|---|
| **DM-draft**（draft k=5 token） | 12.89 ms | **69.1%** |
| **TM-verify**（target 验证） | 5.76 ms | 30.9% |

**整样本 decode 段**（172 轮 + 1 次 DM-prefill）= 3222 ms

| 阶段 | 累计耗时 | 占比 |
|---|---|---|
| DM-prefill（context KV，一次） | 6.7 ms | 0.2% |
| **DM-draft 累计（172 轮）** | 2222 ms | **69.0%** |
| **TM-verify 累计（172 轮）** | 994 ms | 30.8% |

> DM 合计（prefill+draft）69.2% vs TM-verify 30.8%。

**关键观察**：单轮里 DM-draft 就比 TM-verify 贵 2~3 倍（73.9% / 69.1%），且因每轮占比恒定，整样本累计后比例不变。投机解码本想用「便宜的 draft 换掉一次 target 前向」，但视频长 context 下 **draft 反而成了 decode 段最贵的部分**——这是接受率不低却依然亏损的直接原因。

### 10.4 收益拆解

- **整体 speedup = 5574 / 6345 = 0.879**（< 1，反而变慢）。
- **decode 段单独看也是亏的**：baseline 的 decode 段 3000 ms → SD 的 (DM-prefill + TM-verify + DM-draft) = 3771 ms，**decode 段 speedup = 0.796**。投机解码在 decode 段本身就净亏，根本没机会去抵消 prefill。
- **两个独立的亏损来源**：
  1. **prefill 占基线 46%、原封不动**——投机解码完全不碰 prefill，这是死成本。视频帧多 → prompt 长 → prefill 重。
  2. **DM-draft 占 SD 的 43.8%**——DM 虽小（8 层 2048 hidden），但走 **context cross-attention**：每个 draft step 让 6 个 query token 去 attend 全部 22163 个 context token（target 对整段 prompt 的 hidden state，取第 `[1,10,19,28,37]` 5 层拼接经 `fc(10240→2048)`）。注意力规模 ≈ `Q长度 × K长度` = `6 × 22163`，**随 context 线性增长**（22163 ctx → 16.5ms / 16022 ctx → 12.9ms）。accept_length≈3 省下的 TM 调用次数，被 DM 的长 context draft 开销吃光。

> **对比文本场景**：文本 context 短（几百~两千 token），cross-attention 便宜，parallel drafting「一次出 k 个」的优势明显 → 文本 dflash 能到 1.31~1.36x。视频 context 暴涨到 16k~22k，cross-attention 把 DM 单步成本拉到比 TM decode 还贵约 3 倍 → speedup 掉到 <1。

### 10.5 一句话结论

视频场景下投机解码被**两头夹击**：① prefill 占了近一半且无法加速；② decode 段又因 DM 的长 context cross-attention 而净亏。**增帧（16→25）会让 prefill 和 DM-draft 同时更重，speedup 只会进一步恶化**——这也解释了为什么放开帧数上限后加速比不升反降。

---

## 十一、复现命令

```bash
# 在 specforge 环境导出（下载视频 + 生成索引）
python scripts/export_video_eval_datasets.py
python scripts/export_video_eval_datasets.py --datasets videomme --max-per-dataset 50
python scripts/export_video_eval_datasets.py --datasets longvideobench --max-per-dataset 50 --force
```

### prefill / decode 占比实测（第十节）

```bash
# 在 vllm 环境。先起纯 TM server（不挂 draft，TP=2）：
python -m vllm.entrypoints.openai.api_server \
    --model ./model/Qwen3.5-35B-A3B --port 30021 --dtype bfloat16 \
    --gpu-memory-utilization 0.90 --trust-remote-code --max-model-len 32768 \
    --served-model-name default --tensor-parallel-size 2 \
    --limit-mm-per-prompt '{"image": 64}'

# TM prefill/decode（拿 TTFT 与每 token decode，并打印 prompt_tokens 供 DM 用）
python scripts/profile_tm_video.py --port 30021 \
    --index datasets/eval/videomme_index.jsonl --num-frames 25 --label videomme
python scripts/profile_tm_video.py --port 30021 \
    --index datasets/eval/longvideobench_index.jsonl --num-frames 18 --label longvideobench

# DM prefill/decode（--prompt-len 填上一步打印的 PROMPT_TOKENS）
CUDA_VISIBLE_DEVICES=0 python scripts/profile_dflash_dm.py --prompt-len 22163 --label videomme
CUDA_VISIBLE_DEVICES=0 python scripts/profile_dflash_dm.py --prompt-len 16022 --label longvideobench
```

