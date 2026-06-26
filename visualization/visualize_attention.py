"""
Fixed Attention Map Visualization with quantitative TM vs DM comparison.

Fixes:
1. Percentile clipping instead of max normalization (removes attention sink dominance)
2. Optional: exclude first/last N image tokens (known attention sinks)
3. Adds quantitative metrics: Cosine Sim, JSD, Spearman Correlation, Top-K IoU

Target: Qwen3.5-35B-A3B
Draft: Qwen3.5-35B-A3B-DFlash
"""
import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, "/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/SpecForge/model/Qwen3.5-35B-A3B-DFlash")

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoConfig,
    AutoModel,
)

# ============== Config ==============
TARGET_MODEL_PATH = "/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/SpecForge/model/Qwen3.5-35B-A3B"
DRAFT_MODEL_PATH = "/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/SpecForge/model/Qwen3.5-35B-A3B-DFlash"
IMAGE_PATH = "/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/SpecForge/visualization/input/33458386_164431840124_2.jpg"
OUTPUT_DIR = "/prj/corp/crd/morpheus/lasvegas/china-scratch/ziantan/SpecForge/visualization/golden_retriever"
PROMPT = "Describe this image in detail."

VISION_START_ID = 248053
VISION_END_ID = 248054

TARGET_VIZ = [(0, 3, "layer03_early"), (4, 19, "layer19_mid"), (9, 39, "layer39_late")]
DRAFT_VIZ = [(0, "layer0_early"), (3, "layer3_mid"), (7, "layer7_late")]

# Percentile clipping parameters
CLIP_PERCENTILE_LOW = 2    # clip bottom 2%
CLIP_PERCENTILE_HIGH = 98  # clip top 2% (removes sink outliers)
SINK_TOKENS_EXCLUDE = 4    # exclude first N image tokens from attention (attention sinks)


def overlay_attention(image_np, attn_2d, cmap="jet", alpha=0.5,
                            clip_low=CLIP_PERCENTILE_LOW, clip_high=CLIP_PERCENTILE_HIGH):
    """
    Fixed overlay: uses percentile clipping instead of max normalization.
    This prevents attention sinks from dominating the colormap.
    """
    h, w = image_np.shape[:2]
    mask = cv2.resize(attn_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    # Percentile clipping: robust normalization
    vmin = np.percentile(mask, clip_low)
    vmax = np.percentile(mask, clip_high)
    mask = np.clip(mask, vmin, vmax)
    mask = (mask - vmin) / (vmax - vmin + 1e-8)

    colormap = plt.get_cmap(cmap)
    heatmap = colormap(mask)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    blended = (image_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return blended


def save_overlay(image_np, attn_2d, save_path, title="", cmap="jet"):
    blended = overlay_attention(image_np, attn_2d, cmap=cmap, alpha=0.5)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(blended)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def compute_metrics(attn_a, attn_b, name_a="Target", name_b="Draft"):
    """
    Compute quantitative similarity metrics between two attention maps.
    Both inputs are 2D numpy arrays of the same shape (grid_h x grid_w).
    """
    flat_a = attn_a.flatten()
    flat_b = attn_b.flatten()

    # Normalize to probability distributions for divergence measures
    dist_a = flat_a / (flat_a.sum() + 1e-12)
    dist_b = flat_b / (flat_b.sum() + 1e-12)

    # 1. Cosine Similarity
    cos_sim = np.dot(flat_a, flat_b) / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b) + 1e-12)

    # 2. Jensen-Shannon Divergence (symmetric, bounded [0, 1])
    jsd = jensenshannon(dist_a, dist_b) ** 2  # squared to get divergence (not distance)

    # 3. Spearman Rank Correlation (are same patches ranked similarly?)
    spearman_r, spearman_p = stats.spearmanr(flat_a, flat_b)

    # 4. Pearson Correlation
    pearson_r, pearson_p = stats.pearsonr(flat_a, flat_b)

    # 5. Top-K IoU: overlap of top-20% attended patches
    k = max(1, int(0.2 * len(flat_a)))
    top_a = set(np.argsort(flat_a)[-k:])
    top_b = set(np.argsort(flat_b)[-k:])
    topk_iou = len(top_a & top_b) / len(top_a | top_b)

    # 6. MSE (on normalized distributions)
    mse = np.mean((dist_a - dist_b) ** 2)

    metrics = {
        "cosine_similarity": cos_sim,
        "jensen_shannon_div": jsd,
        "spearman_rank_corr": spearman_r,
        "spearman_pvalue": spearman_p,
        "pearson_corr": pearson_r,
        "pearson_pvalue": pearson_p,
        "top20pct_IoU": topk_iou,
        "mse_normalized": mse,
    }
    return metrics


def get_image_token_range(input_ids):
    start_pos = (input_ids == VISION_START_ID).nonzero(as_tuple=True)[0]
    end_pos = (input_ids == VISION_END_ID).nonzero(as_tuple=True)[0]
    img_start = start_pos[0].item() + 1
    img_end = end_pos[0].item()
    return img_start, img_end


def extract_attention_to_image(attn_weights, img_start, img_end, query_positions,
                               grid_h, grid_w, exclude_sink=True):
    """
    Extract and process attention from query_positions to image tokens.
    attn_weights: [heads, q_len, kv_len]

    If exclude_sink=True, zero-out the first SINK_TOKENS_EXCLUDE image tokens
    before averaging (they are attention sinks, not semantically meaningful).
    """
    n_img = img_end - img_start
    attn_to_img = attn_weights[:, query_positions, img_start:img_end]  # [heads, n_q, n_img]

    if exclude_sink and SINK_TOKENS_EXCLUDE > 0:
        attn_to_img[:, :, :SINK_TOKENS_EXCLUDE] = 0.0

    avg = attn_to_img.mean(axis=(0, 1))  # [n_img]
    return avg.reshape(grid_h, grid_w)


def main():
    target_dir = os.path.join(OUTPUT_DIR, "target_attention")
    draft_dir = os.path.join(OUTPUT_DIR, "draft_attention")
    raw_dir = os.path.join(OUTPUT_DIR, "raw_attention_npy")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(draft_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # Load image
    image = Image.open(IMAGE_PATH).convert("RGB")
    image_np = np.array(image)
    print(f"Image size: {image.size} (WxH)")

    # Prepare inputs
    processor = AutoProcessor.from_pretrained(TARGET_MODEL_PATH)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"][0]
    seq_len = input_ids.shape[0]
    img_start, img_end = get_image_token_range(input_ids)
    n_img_tokens = img_end - img_start
    print(f"Sequence length: {seq_len}")
    print(f"Image tokens: [{img_start}, {img_end}), count={n_img_tokens}")

    # Determine grid shape
    import math
    w, h = image.size
    grid_w = math.ceil(w / 32)
    grid_h = math.ceil(h / 32)
    expected = grid_w * grid_h
    print(f"Expected grid: {grid_h}x{grid_w} = {expected}, actual tokens: {n_img_tokens}")

    if expected != n_img_tokens:
        best_h, best_w = grid_h, grid_w
        for gh in range(1, 60):
            if n_img_tokens % gh == 0:
                gw = n_img_tokens // gh
                aspect_diff = abs(gw / gh - w / h)
                if aspect_diff < 0.5:
                    best_h, best_w = gh, gw
                    break
        grid_h, grid_w = best_h, best_w
        print(f"Adjusted grid: {grid_h}x{grid_w} = {grid_h * grid_w}")

    assert grid_h * grid_w == n_img_tokens, f"Cannot match grid to token count"

    text_start = img_end + 1
    text_positions = list(range(text_start, seq_len))
    print(f"Text tokens: [{text_start}, {seq_len}), count={len(text_positions)}")

    # ==================== TARGET MODEL ====================
    print("\n[TARGET] Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        TARGET_MODEL_PATH, dtype=torch.bfloat16, device_map="balanced",
        attn_implementation="eager",
    )
    model.eval()
    inputs_gpu = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    target_layer_ids = [1, 10, 19, 28, 37]
    hidden_collector = {}

    def make_hook(lid):
        def fn(module, inp, out):
            hidden_collector[lid] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
        return fn

    text_model = model.model.language_model
    hooks = [text_model.layers[lid].register_forward_hook(make_hook(lid)) for lid in target_layer_ids]

    print("[TARGET] Running forward...")
    with torch.no_grad():
        outputs = model(**inputs_gpu, output_attentions=True)

    for hk in hooks:
        hk.remove()

    # Extract target attention
    all_attentions = outputs.attentions
    target_img_attns = {}
    target_img_attns_raw = {}  # without sink removal, for comparison
    for attn_idx, real_layer, label in TARGET_VIZ:
        attn = all_attentions[attn_idx][0].float().cpu().numpy()  # [heads, seq, seq]
        # Fixed: with sink exclusion
        attn_map = extract_attention_to_image(
            attn, img_start, img_end, text_positions, grid_h, grid_w, exclude_sink=True)
        target_img_attns[label] = attn_map
        # Raw: without sink exclusion (for debugging)
        attn_map_raw = extract_attention_to_image(
            attn, img_start, img_end, text_positions, grid_h, grid_w, exclude_sink=False)
        target_img_attns_raw[label] = attn_map_raw
        np.save(os.path.join(raw_dir, f"target_{label}.npy"), attn_map)
        print(f"  Layer {real_layer}: max={attn_map.max():.6f}, mean={attn_map.mean():.6f}, "
              f"std={attn_map.std():.6f}")

    # Save target overlays (fixed)
    for label, attn_2d in target_img_attns.items():
        save_path = os.path.join(target_dir, f"{label}.png")
        real_layer = label.split("_")[0].replace("layer", "")
        save_overlay(image_np, attn_2d, save_path,
                           title=f"Target Model — Layer {real_layer} (text→image, percentile-normed)")
        print(f"  Saved: target_attention/{label}.png")

    # Keep hidden states for draft
    target_hidden_cat = torch.cat([hidden_collector[lid] for lid in target_layer_ids], dim=-1)
    embed_tokens = text_model.embed_tokens

    del model, outputs, all_attentions
    torch.cuda.empty_cache()

    # ==================== DRAFT MODEL ====================
    print("\n[DRAFT] Loading model...")
    draft_config = AutoConfig.from_pretrained(DRAFT_MODEL_PATH, trust_remote_code=True)
    draft_model = AutoModel.from_pretrained(
        DRAFT_MODEL_PATH, config=draft_config, dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda().eval()

    block_size = draft_config.block_size
    mask_token_id = draft_config.dflash_config["mask_token_id"]
    anchor_pos = max(0, seq_len - block_size - 1)

    noise_ids = torch.full((1, block_size), mask_token_id, dtype=torch.long, device="cuda")
    noise_ids[0, 0] = input_ids[anchor_pos].to("cuda")
    noise_embedding = embed_tokens(noise_ids.to(embed_tokens.weight.device)).to(torch.bfloat16).cuda()

    target_hidden_for_draft = draft_model.hidden_norm(
        draft_model.fc(target_hidden_cat.cuda().to(torch.bfloat16))
    )

    ctx_len = seq_len
    context_pos = torch.arange(ctx_len, device="cuda").unsqueeze(0)
    draft_pos = torch.arange(anchor_pos, anchor_pos + block_size, device="cuda").unsqueeze(0)
    full_pos = torch.cat([context_pos, draft_pos], dim=1)

    q_len = block_size
    kv_len = ctx_len + block_size
    attn_mask = torch.zeros(1, 1, q_len, kv_len, dtype=torch.bfloat16, device="cuda")
    causal = torch.arange(q_len, device="cuda").unsqueeze(1) >= torch.arange(q_len, device="cuda").unsqueeze(0)
    attn_mask[:, :, :, ctx_len:] = torch.where(causal, 0.0, -torch.inf).to(torch.bfloat16)

    from dflash import apply_rotary_pos_emb as draft_rope

    hidden = noise_embedding
    target_h = target_hidden_for_draft
    position_embeddings = draft_model.rotary_emb(hidden, full_pos)

    draft_all_attns = {}
    for layer_idx, layer in enumerate(draft_model.layers):
        residual = hidden
        h = layer.input_layernorm(hidden)
        attn = layer.self_attn
        bsz, q_l = h.shape[:2]
        ctx_l = target_h.shape[1]

        q = attn.q_proj(h).view(bsz, q_l, -1, attn.head_dim)
        q = attn.q_norm(q).transpose(1, 2)
        k_ctx = attn.k_proj(target_h)
        k_noise = attn.k_proj(h)
        v_ctx = attn.v_proj(target_h)
        v_noise = attn.v_proj(h)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_l + q_l, -1, attn.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_l + q_l, -1, attn.head_dim)
        k = attn.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = draft_rope(q, k, cos, sin)

        num_kv_groups = attn.num_key_value_groups
        if num_kv_groups > 1:
            k = k.repeat_interleave(num_kv_groups, dim=1)
            v = v.repeat_interleave(num_kv_groups, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * attn.scaling
        attn_weights = attn_weights + attn_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        draft_all_attns[layer_idx] = attn_weights[0].detach().cpu().numpy()

        attn_output = torch.matmul(attn_weights.to(v.dtype), v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_l, -1)
        attn_output = attn.o_proj(attn_output)
        hidden = residual + attn_output
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + hidden

    # Extract draft attention
    draft_img_attns = {}
    for layer_idx, label in DRAFT_VIZ:
        attn = draft_all_attns[layer_idx]  # [heads, block_size, kv_len]
        query_positions = list(range(block_size))
        attn_map = extract_attention_to_image(
            attn, img_start, img_end, query_positions, grid_h, grid_w, exclude_sink=True)
        draft_img_attns[label] = attn_map
        np.save(os.path.join(raw_dir, f"draft_{label}.npy"), attn_map)
        print(f"  Layer {layer_idx}: max={attn_map.max():.6f}, mean={attn_map.mean():.6f}, "
              f"std={attn_map.std():.6f}")

    # Save draft overlays (fixed)
    for label, attn_2d in draft_img_attns.items():
        save_path = os.path.join(draft_dir, f"{label}.png")
        real_layer = label.split("_")[0].replace("layer", "")
        save_overlay(image_np, attn_2d, save_path,
                           title=f"Draft (DFlash) — Layer {real_layer} (block→image, percentile-normed)")
        print(f"  Saved: draft_attention/{label}.png")

    del draft_model, draft_all_attns
    torch.cuda.empty_cache()

    # ==================== QUANTITATIVE COMPARISON ====================
    print("\n" + "=" * 60)
    print("QUANTITATIVE COMPARISON: Target Model vs Draft Model")
    print("=" * 60)

    # Match layers for comparison:
    # Target early (layer3) vs Draft early (layer0)
    # Target mid (layer19) vs Draft mid (layer3)
    # Target late (layer39) vs Draft late (layer7)
    comparisons = [
        ("layer03_early", "layer0_early", "Early"),
        ("layer19_mid", "layer3_mid", "Mid"),
        ("layer39_late", "layer7_late", "Late"),
    ]

    all_metrics = {}
    print(f"\n{'Stage':<8} {'CosSim':<10} {'JSD':<10} {'Spearman':<10} {'Pearson':<10} {'Top20%IoU':<12} {'MSE':<12}")
    print("-" * 72)

    for target_label, draft_label, stage in comparisons:
        t_attn = target_img_attns[target_label]
        d_attn = draft_img_attns[draft_label]
        metrics = compute_metrics(t_attn, d_attn)
        all_metrics[stage] = metrics
        print(f"{stage:<8} {metrics['cosine_similarity']:<10.4f} "
              f"{metrics['jensen_shannon_div']:<10.6f} "
              f"{metrics['spearman_rank_corr']:<10.4f} "
              f"{metrics['pearson_corr']:<10.4f} "
              f"{metrics['top20pct_IoU']:<12.4f} "
              f"{metrics['mse_normalized']:<12.2e}")

    # Overall average
    avg_cos = np.mean([m["cosine_similarity"] for m in all_metrics.values()])
    avg_jsd = np.mean([m["jensen_shannon_div"] for m in all_metrics.values()])
    avg_spearman = np.mean([m["spearman_rank_corr"] for m in all_metrics.values()])
    avg_pearson = np.mean([m["pearson_corr"] for m in all_metrics.values()])
    avg_iou = np.mean([m["top20pct_IoU"] for m in all_metrics.values()])
    avg_mse = np.mean([m["mse_normalized"] for m in all_metrics.values()])
    print("-" * 72)
    print(f"{'AVG':<8} {avg_cos:<10.4f} {avg_jsd:<10.6f} {avg_spearman:<10.4f} "
          f"{avg_pearson:<10.4f} {avg_iou:<12.4f} {avg_mse:<12.2e}")

    # ==================== COMPARISON FIGURE ====================
    print("\n[COMPARISON] Generating fixed comparison figure...")
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # Row 0: Original + Target layers
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    for col, (label, attn_2d) in enumerate(target_img_attns.items()):
        blended = overlay_attention(image_np, attn_2d, cmap="jet", alpha=0.5)
        ax = axes[0, col + 1]
        ax.imshow(blended)
        real_layer = label.split("_")[0].replace("layer", "")
        ax.set_title(f"Target Layer {real_layer}", fontsize=11, fontweight="bold")
        ax.axis("off")

    # Row 1: Original + Draft layers
    axes[1, 0].imshow(image_np)
    axes[1, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    for col, (label, attn_2d) in enumerate(draft_img_attns.items()):
        blended = overlay_attention(image_np, attn_2d, cmap="jet", alpha=0.5)
        ax = axes[1, col + 1]
        ax.imshow(blended)
        real_layer = label.split("_")[0].replace("layer", "")
        ax.set_title(f"Draft Layer {real_layer}", fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.text(0.02, 0.75, "Target\n(35B-A3B)", fontsize=13, fontweight="bold",
             va="center", ha="center", rotation=90, color="navy")
    fig.text(0.02, 0.28, "Draft\n(DFlash)", fontsize=13, fontweight="bold",
             va="center", ha="center", rotation=90, color="darkred")

    plt.suptitle(
        "Attention to Image Patches (Fixed: Percentile Clipping + Sink Removal)\n"
        f"Avg Cosine Sim: {avg_cos:.4f} | Avg JSD: {avg_jsd:.6f} | Avg Spearman: {avg_spearman:.4f}",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0.03, 0, 1, 0.93])
    comp_path = os.path.join(OUTPUT_DIR, "comparison.png")
    plt.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: comparison.png")

    # ==================== METRICS FIGURE ====================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    stages = [s for _, _, s in comparisons]

    # Bar charts for key metrics
    cos_vals = [all_metrics[s]["cosine_similarity"] for s in stages]
    spearman_vals = [all_metrics[s]["spearman_rank_corr"] for s in stages]
    iou_vals = [all_metrics[s]["top20pct_IoU"] for s in stages]

    x = np.arange(len(stages))
    width = 0.5

    axes[0].bar(x, cos_vals, width, color=["#2196F3", "#4CAF50", "#FF9800"])
    axes[0].set_title("Cosine Similarity\n(1.0 = identical)", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stages)
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    for i, v in enumerate(cos_vals):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    axes[1].bar(x, spearman_vals, width, color=["#2196F3", "#4CAF50", "#FF9800"])
    axes[1].set_title("Spearman Rank Correlation\n(1.0 = same ranking)", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stages)
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    for i, v in enumerate(spearman_vals):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    axes[2].bar(x, iou_vals, width, color=["#2196F3", "#4CAF50", "#FF9800"])
    axes[2].set_title("Top-20% Patch IoU\n(1.0 = same patches attended)", fontsize=11, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(stages)
    axes[2].set_ylim(0, 1.05)
    axes[2].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    for i, v in enumerate(iou_vals):
        axes[2].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    plt.suptitle("Target vs Draft: Attention Similarity Metrics by Layer Stage",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_comparison.png")
    plt.savefig(metrics_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: metrics_comparison.png")

    # ==================== DIFFERENCE MAP ====================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for col, (target_label, draft_label, stage) in enumerate(comparisons):
        t_attn = target_img_attns[target_label]
        d_attn = draft_img_attns[draft_label]
        # Normalize both to [0, 1] for fair comparison
        t_norm = (t_attn - t_attn.min()) / (t_attn.max() - t_attn.min() + 1e-8)
        d_norm = (d_attn - d_attn.min()) / (d_attn.max() - d_attn.min() + 1e-8)
        diff = np.abs(t_norm - d_norm)

        h_img, w_img = image_np.shape[:2]
        diff_resized = cv2.resize(diff.astype(np.float32), (w_img, h_img))
        diff_resized = diff_resized / (diff_resized.max() + 1e-8)

        colormap = plt.get_cmap("hot")
        heatmap = colormap(diff_resized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        blended = (image_np * 0.5 + heatmap * 0.5).astype(np.uint8)

        axes[col].imshow(blended)
        axes[col].set_title(f"{stage}: |Target - Draft|\nMSE={all_metrics[stage]['mse_normalized']:.2e}",
                            fontsize=11, fontweight="bold")
        axes[col].axis("off")

    plt.suptitle("Attention Difference Maps (brighter = more disagreement)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    diff_path = os.path.join(OUTPUT_DIR, "difference_maps.png")
    plt.savefig(diff_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: difference_maps.png")

    # ==================== SUMMARY ====================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Fix applied: Percentile clipping [{CLIP_PERCENTILE_LOW}%, {CLIP_PERCENTILE_HIGH}%] + "
          f"Sink token exclusion (first {SINK_TOKENS_EXCLUDE} tokens)")
    print(f"\nAvg Cosine Similarity:    {avg_cos:.4f}")
    print(f"Avg JSD:                  {avg_jsd:.6f}")
    print(f"Avg Spearman Correlation: {avg_spearman:.4f}")
    print(f"Avg Top-20% IoU:          {avg_iou:.4f}")
    print(f"\nInterpretation:")
    if avg_cos > 0.9:
        print("  → Draft model attention is VERY SIMILAR to target (cos > 0.9)")
    elif avg_cos > 0.7:
        print("  → Draft model attention is MODERATELY SIMILAR to target (0.7 < cos < 0.9)")
    elif avg_cos > 0.5:
        print("  → Draft model attention has SOME SIMILARITY to target (0.5 < cos < 0.7)")
    else:
        print("  → Draft model attention is QUITE DIFFERENT from target (cos < 0.5)")

    print(f"\nAll outputs in: {OUTPUT_DIR}/")
    print("  comparison.png       - Side-by-side (fixed normalization)")
    print("  metrics_comparison.png     - Quantitative bar charts")
    print("  difference_maps.png        - Per-patch difference overlay")
    print("  target_attention/    - Individual target layers")
    print("  draft_attention/     - Individual draft layers")
    print("  raw_attention_npy/         - Raw numpy arrays for further analysis")


if __name__ == "__main__":
    with torch.no_grad():
        main()
