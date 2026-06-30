from typing import Callable, Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    # NOTE: PerceiverResamplerLayer is intentionally NOT listed here. The video
    # adapter's compress() has data-dependent branches (visual-token count,
    # contiguity) that change how often the resampler runs per sample. If the
    # resampler were its own FSDP unit, those differences would make ranks emit
    # different numbers/sizes of all-gather collectives in the same step, which
    # eventually desyncs the process group and deadlocks (NCCL watchdog timeout).
    # Keeping it folded into the parent FSDP unit removes that desync source; the
    # resampler is tiny (a couple of layers) so there's no overlap/memory cost.
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        self.projector_type = dflash_config.get("projector_type", None)
        self.pure_draft_prefix_len = dflash_config.get("pure_draft_prefix_len", 0)
        self.shift_label = dflash_config.get("shift_label", False)

        if self.projector_type == "domino":
            self.emb_dim = dflash_config["emb_dim"]
            self.gru_hidden_dim = dflash_config["gru_hidden_dim"]
            self.prefix_gru = nn.GRU(
                input_size=config.hidden_size,
                hidden_size=self.gru_hidden_dim,
                num_layers=1,
                batch_first=True,
                bias=False,
            )
            in_dim = config.hidden_size + self.gru_hidden_dim
            self.embed_proj = nn.Sequential(
                nn.Linear(in_dim, self.emb_dim, bias=False),
                nn.SiLU(),
                nn.Linear(self.emb_dim, config.vocab_size, bias=False),
            )
        elif self.projector_type is not None:
            raise ValueError(f"Unknown draft projector_type: {self.projector_type}")

        va_cfg = dflash_config.get("video_adapter")
        if va_cfg and va_cfg.get("enabled", False):
            from .video_adapter import ElasticVideoAdapter

            self.video_adapter = ElasticVideoAdapter(config.hidden_size, va_cfg)
        else:
            self.video_adapter = None

        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        input_ids: Optional[torch.Tensor] = None,
        skip_adapter: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        if not skip_adapter:
            target_hidden = self.hidden_norm(self.fc(target_hidden))

            compressed_visual = None
            if self.video_adapter is not None and input_ids is not None:
                target_hidden, compressed_visual, _ = (
                    self.video_adapter.compress(target_hidden, input_ids)
                )
                if compressed_visual is not None:
                    hidden_states = self.video_adapter.fuse_mask(
                        hidden_states, compressed_visual
                    )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        disable_adapter: bool = False,
        embed_tokens: Optional[nn.Module] = None,
        lm_head: Optional[nn.Module] = None,
    ):
        """Speculative decoding loop.

        VLM support (optional, all default to the text-only behavior):
          pixel_values / image_grid_thw / mm_token_type_ids: vision features fed
            to the target prefill so image placeholder tokens get real features.
            Build them with the target processor's image_processor (see
            scripts/eval_dflash_spec_video.py / generate_dflash_data).
          disable_adapter: when True, skip the video_adapter compress/fuse path
            entirely (control run to measure the adapter's effect on acceptance).
          embed_tokens / lm_head: explicit module refs for nested VLM structures
            (e.g. target.model.language_model.embed_tokens). When None, fall back
            to target.model.embed_tokens / target.lm_head (plain text models).
        """
        self.eval()

        use_adapter = self.video_adapter is not None and not disable_adapter
        if embed_tokens is None:
            embed_tokens = target.model.embed_tokens
        if lm_head is None:
            lm_head = target.lm_head

        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=target.device
        ).unsqueeze(0)

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # Prefill stage
        prefill_kwargs = dict(
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        # VLM: feed vision features so image placeholder tokens get real features.
        if pixel_values is not None:
            prefill_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            prefill_kwargs["image_grid_thw"] = image_grid_thw
        if mm_token_type_ids is not None:
            prefill_kwargs["mm_token_type_ids"] = mm_token_type_ids
        output = target(input_ids, **prefill_kwargs)

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )

        # Video adapter: compress prefill target_hidden for draft KV.
        # disable_adapter (use_adapter=False) gives a control run: the draft KV
        # keeps the full (uncompressed) visual context, equivalent to plain dflash.
        self._spec_generate_kv_len_before = target_hidden.shape[1]
        # ctx_position_ids: the position ids of the draft-KV *context* (the keys
        # before the speculative block). When the adapter compresses the visual
        # tokens, the context shrinks from num_input_tokens to S_compressed, so the
        # contiguous arange no longer matches — mirror training and remap via
        # build_compressed_position_ids (see core/dflash.py:_forward compressed path).
        ctx_position_ids = position_ids[:, :num_input_tokens]
        if use_adapter:
            target_hidden = self.hidden_norm(self.fc(target_hidden))
            target_hidden, _, compression_info = self.video_adapter.compress(
                target_hidden, input_ids
            )
            if compression_info is not None:
                ctx_position_ids = self.video_adapter.build_compressed_position_ids(
                    input_ids, compression_info
                )
            self._spec_generate_skip_fc = True
        else:
            self._spec_generate_skip_fc = False
        self._spec_generate_kv_len_after = target_hidden.shape[1]

        # Decode stage
        acceptance_lengths = []
        start = input_ids.shape[1]
        # Draft-KV context length in (possibly compressed) cache coords. In off mode
        # this equals num_input_tokens; with the adapter it is S_compressed.
        prefill_ctx_len = target_hidden.shape[1]
        # prev_start: original-coord start of the previous block (the accepted span
        # fed back as context next iter). first_iter feeds the prefill context.
        prev_start = start
        first_iter = True
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = embed_tokens(block_output_ids)
            # Position ids for this forward = [context positions ; block positions].
            # First iter feeds the (possibly compressed) prefill context; later iters
            # feed the just-accepted decode tokens (plain text, original coords).
            if first_iter:
                ctx_pos = ctx_position_ids
            else:
                ctx_pos = position_ids[:, prev_start:start]
            fwd_position_ids = torch.cat([ctx_pos, block_position_ids], dim=1)
            draft_logits = lm_head(
                self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=fwd_position_ids,
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                    skip_adapter=self._spec_generate_skip_fc,
                )[:, -block_size + 1 :, :]
            )
            # Crop the draft KV back to the context boundary (drop this block's keys).
            # In compressed coords: prefill_ctx_len + (accepted decode tokens so far).
            past_key_values_draft.crop(prefill_ctx_len + (start - num_input_tokens))
            block_output_ids[:, 1:] = sample(draft_logits)

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            # Next iter's context = the tokens accepted this round (orig coords
            # [start : start+accept+1]); remember this start to slice their positions.
            prev_start = start
            first_iter = False
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )[:, : acceptance_length + 1, :]

            # Decode-phase target_hidden is pure text; still needs fc+norm when
            # adapter was used for prefill (skip_adapter covers both fc and adapter)
            if self._spec_generate_skip_fc:
                target_hidden = self.hidden_norm(self.fc(target_hidden))

            acceptance_lengths.append(acceptance_length + 1)
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        # Expose decode stats for evaluation (read by the caller after the call).
        # acceptance_lengths[i] = accepted tokens in block i (incl. the bonus token).
        self._spec_generate_acceptance_lengths = acceptance_lengths

        return output_ids
