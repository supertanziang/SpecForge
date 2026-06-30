from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
import os
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather
from transformers import AutoModelForCausalLM

from specforge.distributed import get_tp_group

from .sglang_backend import SGLangRunner


@dataclass
class DFlashTargetOutput:
    hidden_states: torch.Tensor  # [batch, seq_len, hidden_size]
    input_ids: torch.Tensor  # [batch, seq_len]
    attention_mask: torch.Tensor  # [batch, seq_len]
    loss_mask: torch.Tensor  # [batch, seq_len]
    visual_token_mask: Optional[torch.Tensor] = None  # [batch, seq_len], bool


class DFlashTargetModel(ABC):
    """
    Abstract base class for DFlash target model backend.
    """

    def __init__(self):
        self.capture_layer_ids = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "DFlashTargetModel":
        """Initialize the target model backend."""

    @abstractmethod
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        image_files: Optional[List[str]] = None,
    ) -> DFlashTargetOutput:
        """Generate context hidden states for DFlash training.

        image_files: list of frame paths (single image = 1-element list, video =
        N frames). None for plain-text samples.
        """

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        """Set which layers' hidden states to capture."""
        self.capture_layer_ids = layer_ids


class SGLangDFlashTargetModel(DFlashTargetModel):
    def __init__(self, model_runner: SGLangRunner):
        super().__init__()
        self.model_runner = model_runner

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "SGLangDFlashTargetModel":
        tp_size = dist.get_world_size(get_tp_group())
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            enable_return_hidden_states=True,  # Critical for DFlash
            disable_cuda_graph=True,
            tp_size=tp_size,
            pp_size=1,
            **kwargs,
        )

        tp_rank = dist.get_rank(get_tp_group())
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
        model_config = ModelConfig.from_server_args(server_args)

        model_runner = SGLangRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=torch.cuda.current_device(),
            tp_rank=dist.get_rank(get_tp_group()),
            tp_size=server_args.tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            pp_rank=0,
            pp_size=1,
            server_args=server_args,
            nccl_port=None,
        )
        return cls(model_runner)

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        super().set_capture_layers(layer_ids)
        if hasattr(self.model_runner.model, "set_eagle3_layers_to_capture"):
            self.model_runner.model.set_eagle3_layers_to_capture(layer_ids)
            print(self.model_runner.model.model.layers_to_capture)

    @torch.no_grad
    def _extend(self, reqs):
        cache_params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.model_runner.server_args.page_size,
        )
        tree_cache = RadixCache(cache_params)

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()

        if require_mlp_sync(self.model_runner.server_args):
            Scheduler.prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.model_runner.server_args.dp_size,
                attn_tp_size=1,
                tp_group=self.model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                speculative_num_draft_tokens=None,
                require_mlp_tp_gather=require_mlp_tp_gather(
                    self.model_runner.server_args
                ),
                disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
                offload_tags=set(),
            )

        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        output = self.model_runner.forward(forward_batch)
        if hasattr(output, "logits_output"):
            output = output.logits_output

        input_lens = [len(req.origin_input_ids) for req in reqs]
        if (
            hasattr(output, "aux_hidden_states")
            and output.aux_hidden_states is not None
        ):
            hidden_states_list = torch.split(
                output.aux_hidden_states, input_lens, dim=0
            )
        elif hasattr(output, "hidden_states") and output.hidden_states is not None:
            hidden_states_list = torch.split(output.hidden_states, input_lens, dim=0)
        else:
            raise ValueError("SGLang output does not contain hidden states.")

        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

        return hidden_states_list

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        image_files: Optional[List[str]] = None,
    ) -> DFlashTargetOutput:
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
        reqs, data_cache = [], []

        if isinstance(input_ids, torch.Tensor):
            input_ids_list = torch.split(input_ids, 1, dim=0)
            attn_mask_list = torch.split(attention_mask, 1, dim=0)
            loss_mask_list = torch.split(loss_mask, 1, dim=0)

        for idx, (curr_ids, curr_attn, curr_loss) in enumerate(
            zip(input_ids_list, attn_mask_list, loss_mask_list)
        ):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=curr_ids.view(-1).tolist(),
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            data_cache.append((curr_ids, curr_attn, curr_loss))
            reqs.append(req)

        hidden_states_list = self._extend(reqs)

        # Stack back to batch
        hidden_states = torch.cat([h.unsqueeze(0) for h in hidden_states_list], dim=0)
        input_ids = torch.cat([d[0] for d in data_cache], dim=0)
        attention_mask = torch.cat([d[1] for d in data_cache], dim=0)
        loss_mask = torch.cat([d[2] for d in data_cache], dim=0)

        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )


class HFDFlashTargetModel(DFlashTargetModel):
    def __init__(self, model: nn.Module, is_vlm: bool = False, processor=None):
        super().__init__()
        self.model = model
        self.is_vlm = is_vlm
        self.processor = processor

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        is_vlm: bool = False,
        processor=None,
        **kwargs,
    ) -> "HFDFlashTargetModel":

        if is_vlm:
            # VLM target (e.g. Qwen2.5-VL, Qwen3.5-VL): load the image-text-to-text
            # model so the vision tower runs and image features get injected into the
            # hidden states. AutoModelForImageTextToText resolves the right class from
            # the model config (Qwen2_5_VLForConditionalGeneration /
            # Qwen3_5MoeForConditionalGeneration / ...), so this stays model-agnostic.
            from transformers import AutoModelForImageTextToText

            # Optional accelerate-style sharding within this rank (e.g. 35B-VL spans
            # multiple GPUs per training rank). When SPECFORGE_TARGET_DEVICE_MAP is
            # set, we hand it to from_pretrained and skip the explicit .to(device)
            # below — accelerate has already placed every shard.
            device_map_env = os.environ.get("SPECFORGE_TARGET_DEVICE_MAP")
            extra_kwargs = dict(kwargs)
            sharded = False
            if device_map_env:
                # device_map="auto"/"balanced"/"sequential": accelerate picks placement
                # from CUDA_VISIBLE_DEVICES (the launcher restricts each rank to its
                # own slice of physical GPUs).
                extra_kwargs["device_map"] = device_map_env
                sharded = True

            target_model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                **extra_kwargs,
            ).eval()
        else:
            target_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                output_hidden_states=True,
                trust_remote_code=trust_remote_code,
                **kwargs,
            ).eval()
            sharded = False

        if device and not sharded:
            target_model = target_model.to(device)

        return cls(target_model, is_vlm=is_vlm, processor=processor)

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        image_files: Optional[List[str]] = None,
    ) -> DFlashTargetOutput:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
        }

        if self.is_vlm and image_files:
            # input_ids already contains the expanded image placeholder tokens
            # (produced offline). We only need the vision features here, so run the
            # image_processor alone (no second text/image expansion). Mirrors ViSpec's
            # build_target_hidden.
            #
            # Video samples are stored as N extracted frames (image_files of len N);
            # a single image is the N=1 case. We feed all frames to image_processor as
            # a multi-image batch, matching how the eval side (benchmark_sd_vlm.py)
            # treats a video: N independent image_url blocks, NOT a native video input.
            # So pixel_values/image_grid_thw (shape [N,3]) use image_token_id throughout
            # -- no video_token / pixel_values_videos / video_grid_thw involved.
            #
            # When image_files is empty/None (plain-text sample fed to a VLM target via
            # --vlm-text-data), we skip vision processing entirely: Qwen2.5-VL's
            # forward runs text-only when pixel_values/image_grid_thw are absent.
            assert self.processor is not None, "VLM target requires a processor"
            images = [Image.open(p).convert("RGB") for p in image_files]
            img_proc = self.processor.image_processor(
                images=images, return_tensors="pt"
            ).to(input_ids.device)
            if "pixel_values" in img_proc:
                model_inputs["pixel_values"] = img_proc["pixel_values"]
            if "image_grid_thw" in img_proc:
                model_inputs["image_grid_thw"] = img_proc["image_grid_thw"]

            # Guard against offline/online tokenize mismatch: the number of image
            # placeholder tokens in input_ids must equal the total vision tokens the
            # processor emits for these frames (sum of grid t*h*w / merge_size^2).
            # A mismatch (e.g. different min/max_pixels than at generation time, or
            # wrong frame count) would otherwise blow up inside the model forward with
            # an opaque shape error.
            cfg = self.model.config
            image_token_id = getattr(cfg, "image_token_id", None)
            if "image_grid_thw" in img_proc and image_token_id is not None:
                merge_size = getattr(self.processor.image_processor, "merge_size", 2)
                grid = img_proc["image_grid_thw"]
                n_vision = int((grid.prod(dim=-1) // (merge_size**2)).sum().item())
                n_placeholder = int((input_ids == image_token_id).sum().item())
                assert n_vision == n_placeholder, (
                    f"vision token count mismatch: processor emits {n_vision} tokens "
                    f"for {len(images)} frame(s) but input_ids has {n_placeholder} "
                    f"image placeholders. Check that min/max_pixels match the values "
                    f"used when generating this sample."
                )

            # Qwen3.5-VL (Qwen3_5MoeForConditionalGeneration) requires mm_token_type_ids
            # for M-RoPE: 0=text, 1=image, 2=video. transformers>=5.8 raises if missing.
            # Older Qwen2.5-VL forward doesn't accept this kwarg; gate by signature.
            # Multi-frame video is fed as images (image_token_id), so we only mark
            # image positions -- never video_token (no video tokens are produced here).
            forward_params = inspect.signature(self.model.forward).parameters
            if "mm_token_type_ids" in forward_params:
                mm_ids = torch.zeros_like(input_ids, dtype=torch.int32)
                if image_token_id is not None:
                    mm_ids[input_ids == image_token_id] = 1
                model_inputs["mm_token_type_ids"] = mm_ids

        outputs = self.model(**model_inputs)

        # hidden_states[0] = embedding output; hidden_states[i+1] = layer i output
        offset = 1
        selected = []
        if self.capture_layer_ids is not None:
            for idx in self.capture_layer_ids:
                # accelerate device_map can place layers on different GPUs;
                # bring all captured hidden_states back to the input device
                # before concatenation.
                selected.append(outputs.hidden_states[idx + offset].to(input_ids.device))
            hidden_states = torch.cat(selected, dim=-1)
        else:
            hidden_states = outputs.hidden_states[-1].to(input_ids.device)

        visual_token_mask = None
        if self.is_vlm:
            image_token_id = getattr(self.model.config, "image_token_id", None)
            if image_token_id is not None:
                visual_token_mask = (input_ids == image_token_id)

        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            visual_token_mask=visual_token_mask,
        )


def get_dflash_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> DFlashTargetModel:
    if backend == "sglang":
        if kwargs.get("is_vlm", False):
            raise ValueError(
                "VLM target is only supported with the 'hf' backend; "
                "the sglang DFlash backend does not return image-aware hidden states."
            )
        # sglang backend does not accept these hf-only kwargs
        kwargs.pop("is_vlm", None)
        kwargs.pop("processor", None)
        return SGLangDFlashTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "hf":
        return HFDFlashTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")
