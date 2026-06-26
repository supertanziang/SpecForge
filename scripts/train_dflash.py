#!/usr/bin/env python3
# coding=utf-8
"""DFlash Training Script."""

import argparse
import functools
import logging
import math
import os
import shutil
import time
import warnings
from typing import Optional, Tuple

# Per-rank GPU slicing: when SPECFORGE_PER_RANK_DEVICES is set (e.g. 2 means each
# torchrun rank gets 2 physical GPUs out of CUDA_VISIBLE_DEVICES), re-slice
# CUDA_VISIBLE_DEVICES BEFORE importing torch so each rank only sees its own
# GPUs as cuda:0..cuda:N-1. Used for VLM target sharding via device_map.
_per_rank = os.environ.get("SPECFORGE_PER_RANK_DEVICES")
if _per_rank:
    _per_rank_n = int(_per_rank)
    _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if _visible is None:
        _all_devs = [str(i) for i in range(_per_rank_n * int(os.environ.get("WORLD_SIZE", "1")))]
    else:
        _all_devs = [d.strip() for d in _visible.split(",") if d.strip()]
    _start = _local_rank * _per_rank_n
    _slice = _all_devs[_start : _start + _per_rank_n]
    assert len(_slice) == _per_rank_n, (
        f"SPECFORGE_PER_RANK_DEVICES={_per_rank_n} but only {len(_slice)} GPUs left "
        f"for local_rank={_local_rank} (visible={_all_devs})"
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(_slice)
    print(f"[per-rank-slice] LOCAL_RANK={_local_rank} -> CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from datasets import load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.dflash import OnlineDFlashModel
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.data.online_mm_dataset import OnlineMMDataset, prepare_mm_dataloader
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.dflash_target_model import (
    DFlashTargetModel,
    get_dflash_target_model,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import get_last_checkpoint, print_on_rank0, print_with_rank


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash Draft Model")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="hf",
        choices=["sglang", "hf"],
        help="Backend for target model: 'sglang' (service) or 'hf' (local)",
    )
    model_group.add_argument("--draft-config-path", type=str, default=None)
    model_group.add_argument("--block-size", type=int, default=16)
    model_group.add_argument("--num-draft-layers", type=int, default=1)
    model_group.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="MASK token ID. If not provided, auto-detect from tokenizer.",
    )
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
        help="Attention backend for draft model.",
    )
    model_group.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    model_group.add_argument(
        "--num-anchors",
        type=int,
        default=512,
        help="Number of anchor positions per sequence",
    )
    model_group.add_argument(
        "--loss-decay-gamma",
        type=float,
        default=None,
        help="Gamma for exponential loss decay weighting (paper Eq.4). "
        "Suggested: 7 for block_size=16, 5 for 10, 4 for 8. None disables.",
    )
    model_group.add_argument(
        "--embedding-key",
        type=str,
        default=None,
        help="Embedding weight key in the target model. "
        "Default: 'model.embed_tokens.weight' for standard models, "
        "'model.language_model.embed_tokens.weight' for multimodal models like Qwen3.5-A3B.",
    )
    model_group.add_argument(
        "--lm-head-key",
        type=str,
        default=None,
        help="LM head weight key in the target model. Default: 'lm_head.weight'.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    vlm_group = parser.add_argument_group("vlm")
    vlm_group.add_argument(
        "--is-vlm",
        action="store_true",
        help="Train against a VLM target (e.g. Qwen2.5-VL). Uses a pre-tokenized "
        "JSONL with input_ids/loss_mask/image_file and an HF VLM backend.",
    )
    vlm_group.add_argument(
        "--vlm-text-data",
        action="store_true",
        help="VLM target (--is-vlm) but feed plain-text ShareGPT conversations "
        "(no images). Routes to build_eagle3_dataset online tokenization instead "
        "of the pre-tokenized OnlineMMDataset. Target forward degrades to text-only "
        "when no image_file is present, so image samples still work if mixed in.",
    )
    vlm_group.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Processor min_pixels (e.g. 256*28*28 for Qwen2.5-VL). MUST match the "
        "value used when the ViSpec full_ids were generated, else the recomputed "
        "image_grid_thw will disagree with the baked-in image placeholder tokens and "
        "forward will crash. Leave unset for processors that don't accept min_pixels "
        "(e.g. Qwen3VLProcessor, which uses size={longest_edge,shortest_edge}).",
    )
    vlm_group.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Processor max_pixels (e.g. 1280*28*28 for Qwen2.5-VL). Must match ViSpec "
        "generation. Leave unset for processors that don't accept max_pixels.",
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument(
        "--init-draft-path",
        type=str,
        default=None,
        help="Path to a draft checkpoint to initialize weights from (e.g. a stage1 "
        "checkpoint), WITHOUT resuming optimizer/scheduler/step state. Use this for "
        "stage2 training in a fresh --output-dir that continues from stage1 weights. "
        "Ignored when --resume finds a checkpoint in --output-dir (resuming the "
        "current run takes precedence).",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=1000)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="The size of the tensor parallel for the target model",
    )

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    # SGLang specific args
    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)

    return parser.parse_args()


def build_models(args, processor=None) -> Tuple[DFlashTargetModel, DFlashDraftModel]:
    """Build target model (backend wrapper) and draft model."""
    print_on_rank0(
        f"Loading target model from {args.target_model_path} using {args.target_model_backend} backend"
    )

    target_model_kwargs = {}
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

    if args.is_vlm:
        if args.target_model_backend != "hf":
            raise ValueError("--is-vlm requires --target-model-backend hf")
        target_model_kwargs["is_vlm"] = True
        target_model_kwargs["processor"] = processor

    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_model_kwargs,
    )

    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(args.draft_config_path)
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
        # Warn if command-line args differ from config
        if (
            hasattr(draft_config, "block_size")
            and draft_config.block_size != args.block_size
        ):
            print_on_rank0(
                f"Warning: checkpoint block_size ({draft_config.block_size}) differs from "
                f"command-line arg ({args.block_size}). Using checkpoint value."
            )
    else:
        target_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config.num_hidden_layers = args.num_draft_layers
        draft_config.block_size = args.block_size
        draft_config.num_target_layers = target_config.num_hidden_layers
        print_on_rank0("Auto-generated draft config from target model")

    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}

    draft_config._attn_implementation = args.attention_backend
    print_on_rank0(f"Using attention backend: {args.attention_backend}")

    draft_model = DFlashDraftModel(draft_config).cuda().to(torch.bfloat16)

    target_model.set_capture_layers(draft_model.target_layer_ids)

    print_on_rank0(
        f"Draft config: block_size={draft_config.block_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"num_target_layers={draft_config.num_target_layers}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )

    return target_model, draft_model


def build_dataloader(args, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train and eval dataloaders."""
    import hashlib

    if args.is_vlm and not args.vlm_text_data:
        # VLM path: pre-tokenized JSONL (input_ids / loss_mask / image_file), bs=1.
        # Target hidden states are computed online in the training loop.
        train_mm_dataset = OnlineMMDataset(args.train_data_path, max_len=args.max_length)
        print_on_rank0(f"Loaded VLM train dataset: {len(train_mm_dataset)} samples")
        train_dataloader = prepare_mm_dataloader(
            train_mm_dataset,
            num_workers=args.dataloader_num_workers,
            shuffle=True,
            process_group=get_dp_group(),
        )
        eval_dataloader = None
        if args.eval_data_path:
            eval_mm_dataset = OnlineMMDataset(
                args.eval_data_path, max_len=args.max_length
            )
            eval_dataloader = prepare_mm_dataloader(
                eval_mm_dataset,
                num_workers=args.dataloader_num_workers,
                shuffle=False,
                process_group=get_dp_group(),
            )
        return train_dataloader, eval_dataloader

    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    train_eagle3_dataset = build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
    )

    min_loss_tokens = 2 * args.block_size
    original_size = len(train_eagle3_dataset)
    train_eagle3_dataset = train_eagle3_dataset.filter(
        lambda x: x["loss_mask"].sum() >= min_loss_tokens
    )
    print_on_rank0(
        f"Filtered train dataset: {original_size} -> {len(train_eagle3_dataset)} samples"
    )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
    )

    eval_dataloader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )

    return train_dataloader, eval_dataloader


def save_checkpoint(args, epoch, step, dflash_model, draft_model, optimizer):
    """Save checkpoint."""
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(dflash_model, StateDictType.FULL_STATE_DICT):
        state_dict = dflash_model.state_dict()
        draft_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }

        if dist.get_rank() == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "args": args,
                    **optimizer.state_dict(),
                },
                os.path.join(save_dir, "training_state.pt"),
            )

            draft_model.save_pretrained(save_dir, state_dict=draft_state_dict)

            modeling_src = os.path.join(
                os.path.dirname(__file__),
                "..",
                "specforge",
                "modeling",
                "draft",
                "dflash.py",
            )
            modeling_dst = os.path.join(save_dir, "dflash.py")
            if os.path.exists(modeling_src):
                shutil.copy(modeling_src, modeling_dst)

            print_on_rank0(f"Saved checkpoint to {save_dir}")

    dist.barrier()


def record_metrics(
    args,
    loss: float,
    accuracy: float,
    global_step: int,
    tracker,
    optimizer,
    train_dataloader=None,
    mode: str = "train",
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()

    logdict[f"{mode}/loss"] = loss
    logdict[f"{mode}/accuracy"] = accuracy

    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step} [{global_step}/{args.num_epochs * len(train_dataloader) // args.accumulation_steps}?], Loss: {loss:.4f}, Acc: {accuracy:.4f}"
    )

    tracker.log(logdict, step=global_step)


def main():

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    args = parse_args()
    set_seed(args.seed)

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed")

    draft_model_last_checkpoint = None
    ckpt_info = (0, 0)
    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)
        print(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    # If resuming, load config from checkpoint to ensure consistency
    if draft_model_last_checkpoint:
        checkpoint_config_path = os.path.join(
            draft_model_last_checkpoint, "config.json"
        )
        if os.path.exists(checkpoint_config_path):
            print(f"Loading draft config from checkpoint: {checkpoint_config_path}")
            args.draft_config_path = checkpoint_config_path

    # Build processor (VLM) / tokenizer up front; target model needs the processor.
    processor = None
    if args.is_vlm:
        # Only pass min/max_pixels when explicitly set: Qwen2.5-VL's processor accepts
        # them, but Qwen3VLProcessor does not (it uses size={longest_edge,shortest_edge}).
        processor_kwargs = {}
        if args.min_pixels is not None:
            processor_kwargs["min_pixels"] = args.min_pixels
        if args.max_pixels is not None:
            processor_kwargs["max_pixels"] = args.max_pixels
        processor = AutoProcessor.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
            **processor_kwargs,
        )
        tokenizer = processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    target_model, draft_model = build_models(args, processor=processor)

    resume_state = None
    if draft_model_last_checkpoint:
        loaded_model = DFlashDraftModel.from_pretrained(
            draft_model_last_checkpoint, torch_dtype=torch.bfloat16
        )
        draft_model.load_state_dict(loaded_model.state_dict())
        del loaded_model
        print("Loaded draft model weights from checkpoint")

        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if os.path.exists(training_state_path):
            resume_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
            print(
                f"Will resume from epoch {resume_state['epoch']}, "
                f"step {resume_state['global_step']}"
            )
    elif args.init_draft_path:
        # stage2: initialize draft weights from a stage1 checkpoint WITHOUT resuming
        # optimizer/scheduler/step state (resume_state stays None, output-dir is fresh).
        loaded_model = DFlashDraftModel.from_pretrained(
            args.init_draft_path, torch_dtype=torch.bfloat16
        )
        draft_model.load_state_dict(loaded_model.state_dict())
        del loaded_model
        print(
            f"Initialized draft weights from {args.init_draft_path} "
            "(fresh optimizer/scheduler/step)"
        )

    if args.mask_token_id is not None:
        mask_token_id = args.mask_token_id
    elif args.is_vlm:
        # VLM target embeddings are frozen and not resized; auto-adding a MASK token
        # would yield an out-of-range id. Require an explicit free-slot token id.
        raise ValueError(
            "--is-vlm requires an explicit --mask-token-id (< vocab_size, a free "
            "embedding slot, e.g. 151657 for Qwen2.5-VL)."
        )
    elif tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    print_on_rank0(f"Using mask_token_id: {mask_token_id}")

    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids
    print_on_rank0(f"dflash_config: {draft_model.config.dflash_config}")

    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")

    print_on_rank0("Loading target embeddings and head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )

    dflash_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
    )

    # Wrap each transformer block as its own FSDP unit so that all-gather /
    # reduce-scatter overlap with compute. Without an auto_wrap_policy the
    # whole model is a single FSDP unit, forcing every collective onto the
    # critical path with no overlap. The block class is resolved from the
    # draft model's `_no_split_modules` so this stays architecture-agnostic
    # rather than hardcoding a specific decoder-layer class.
    fsdp_kwargs = dict(
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    block_names = set(getattr(draft_model, "_no_split_modules", None) or [])
    block_classes = {
        type(m) for m in dflash_model.modules() if type(m).__name__ in block_names
    }
    if block_classes:
        fsdp_kwargs["auto_wrap_policy"] = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=block_classes,
        )
    else:
        print_with_rank(
            "No _no_split_modules on draft model; falling back to single-unit "
            "FSDP wrap (no compute-comm overlap)."
        )
    dflash_model = FSDP(dflash_model, **fsdp_kwargs)
    print_with_rank("Initialized FSDP")

    start_epoch = ckpt_info[0]
    global_step = ckpt_info[1]

    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )

    if resume_state is not None:
        optimizer.scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state
        print_on_rank0(
            f"Restored optimizer/scheduler state: "
            f"epoch={start_epoch}, step={global_step}, "
            f"lr={optimizer.get_learning_rate():.6f}"
        )

    skip_steps = global_step - start_epoch * len(train_dataloader)

    print_on_rank0(f"Initializing tracker (report_to={args.report_to})...")
    tracker = create_tracker(args, args.output_dir)
    print_on_rank0("Tracker initialized successfully.")

    last_time = time.time()
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        draft_model.train()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1

            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()

            _prof = os.environ.get("SPECFORGE_PROFILE_STEPS")
            _do_prof = _prof and global_step <= int(_prof) and dist.get_rank() == 0
            if _do_prof:
                torch.cuda.synchronize()
                _t0 = time.time()
            target_output = target_model.generate_dflash_data(
                input_ids,
                attention_mask,
                loss_mask,
                image_file=data.get("image_file") if args.is_vlm else None,
            )
            hidden_states = target_output.hidden_states.cuda()  # Ensure on GPU
            if _do_prof:
                torch.cuda.synchronize()
                _t1 = time.time()

            loss, accuracy = dflash_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
            )
            if _do_prof:
                torch.cuda.synchronize()
                _t2 = time.time()

            (loss / args.accumulation_steps).backward()
            if _do_prof:
                torch.cuda.synchronize()
                _t3 = time.time()
                print(
                    f"[profile] step={global_step} target_fwd={_t1-_t0:.3f}s "
                    f"draft_fwd+loss={_t2-_t1:.3f}s backward={_t3-_t2:.3f}s "
                    f"total={_t3-_t0:.3f}s",
                    flush=True,
                )

            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                loss_log = loss.clone()
                acc_log = accuracy.clone()
                dist.all_reduce(loss_log)
                dist.all_reduce(acc_log)
                loss_log = loss_log / dist.get_world_size()
                acc_log = acc_log / dist.get_world_size()

                record_metrics(
                    args,
                    loss_log.item(),
                    acc_log.item(),
                    global_step,
                    tracker,
                    optimizer,
                    train_dataloader,
                    mode="train",
                )

            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{accuracy.item():.4f}",
                        "iter_time": f"{elapsed:.2f}s",
                    }
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(
                    args, epoch, global_step, dflash_model, draft_model, optimizer
                )

    save_checkpoint(
        args, args.num_epochs, global_step, dflash_model, draft_model, optimizer
    )

    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
