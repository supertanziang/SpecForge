"""
测 target model (TM) 在真实视频样本上的 prefill / decode 时间, 用 OpenAI 流式接口:
  - TTFT (time-to-first-token) ≈ prefill 段 (vision encode + LLM prefill);
  - (total - TTFT) / (completion_tokens - 1) ≈ 每 token 的 decode 时间;
  - prompt_tokens 由 server 的 usage 返回 (用于喂给 DM profiling 的 --prompt-len)。

仅对 server 不挂 draft 的纯 TM 测量。抽帧逻辑复刻 benchmark_sd_vlm.extract_video_frames。
"""

import argparse
import base64
import io
import json
import time

import cv2
import requests
from PIL import Image


def extract_video_frames(video_path: str, num_frames: int):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total / num_frames) for i in range(num_frames)]
    urls = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        urls.append(f"data:image/jpeg;base64,{b64}")
    cap.release()
    return urls


def run(port, item, num_frames, max_tokens):
    frames = extract_video_frames(item["video_path"], num_frames)
    content = [{"type": "image_url", "image_url": {"url": u}} for u in frames]
    content.append({"type": "text", "text": item["question"]})
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    start = time.perf_counter()
    ttft = None
    n_tok = 0
    prompt_tokens = None
    with requests.post(url, json=payload, stream=True, timeout=1200) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode()
            if not s.startswith("data: "):
                continue
            s = s[6:]
            if s == "[DONE]":
                break
            chunk = json.loads(s)
            if chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    n_tok += 1
            if chunk.get("usage"):
                prompt_tokens = chunk["usage"].get("prompt_tokens")
    total = time.perf_counter() - start
    decode_per_tok = (total - ttft) / (n_tok - 1) if n_tok > 1 else 0
    return {
        "num_frames": num_frames,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": n_tok,
        "ttft_s": ttft,
        "total_s": total,
        "decode_per_tok_ms": decode_per_tok * 1000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30021)
    ap.add_argument("--index", required=True, help="<name>_index.jsonl")
    ap.add_argument("--num-frames", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    item = json.loads(open(args.index).readline())
    # warmup
    run(args.port, item, args.num_frames, max_tokens=32)
    res = run(args.port, item, args.num_frames, args.max_tokens)

    print(f"[TM {args.label}] frames={res['num_frames']} prompt_tokens={res['prompt_tokens']} "
          f"completion={res['completion_tokens']}")
    print(f"  TM-prefill (TTFT):           {res['ttft_s']*1000:.1f} ms")
    print(f"  TM-decode  (每 token):        {res['decode_per_tok_ms']:.3f} ms")
    print(f"  total:                       {res['total_s']*1000:.1f} ms")
    print(f"  PROMPT_TOKENS={res['prompt_tokens']}")  # 供 DM 脚本 --prompt-len 用


if __name__ == "__main__":
    main()
