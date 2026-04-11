import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from .attention import RotaryEmbeddingESM, rekv_attention_forward


def huggingface_forward(forward):
    def hf_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        assert not output_attentions
        ret = forward(
            self, hidden_states, hidden_states,
            position_ids, use_cache, past_key_value,
            self.q_proj, self.k_proj, self.v_proj, self.o_proj, 
            self.head_dim, self.num_heads, self.num_key_value_heads
        )
        if use_cache:
            o, pkv = ret
        else:
            o = ret
            pkv = None

        return o, None, pkv

    return hf_forward


def patch_hf(
    model,
    attn_kwargs: dict = {},
    base = None, 
    distance_scale = None,
    **kwargs
):
    attn_kwargs.update(kwargs)
    # This approach lacks scalability and will be refactored.
    from transformers import LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Qwen2Model
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel, BaseModelOutputWithPast

    def _resolve_model_core(hf_model):
        return hf_model.model if hasattr(hf_model, "model") else hf_model

    def _infer_attention_shape(attn_module, model_core):
        dim_head = getattr(attn_module, "head_dim", None)
        num_heads = getattr(attn_module, "num_heads", None)
        num_heads_kv = getattr(attn_module, "num_key_value_heads", None)
        config = getattr(model_core, "config", None)

        if num_heads is None and config is not None:
            num_heads = getattr(config, "num_attention_heads", None)
        if num_heads_kv is None and config is not None:
            num_heads_kv = getattr(config, "num_key_value_heads", None)
        if dim_head is None and config is not None and num_heads is not None:
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is not None:
                dim_head = int(hidden_size // num_heads)

        if dim_head is None:
            raise ValueError(f"Could not infer attention head dimension for {attn_module.__class__.__name__}.")
        if num_heads is None:
            num_heads = int(attn_module.q_proj.out_features // dim_head)
        if num_heads_kv is None:
            num_heads_kv = int(attn_module.k_proj.out_features // dim_head)

        return int(dim_head), int(num_heads), int(num_heads_kv)

    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        *args,
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if hasattr(self, "config") and hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb

        if use_cache:
            pkv = tuple()

        else:
            pkv = None

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            residual = hidden_states
            hidden_states = decoder_layer.input_layernorm(hidden_states)
            attn_output, _, layer_past_key_value = decoder_layer.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=self.position_bias,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            if use_cache:
                pkv = pkv + (layer_past_key_value,)

            if output_attentions:
                all_self_attns += (None,)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, pkv, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=pkv,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    forward = huggingface_forward(rekv_attention_forward(**attn_kwargs))

    if isinstance(model, (LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Qwen2Model)) or model.__class__.__name__ == "MiniCPMForCausalLM":
        model_core = _resolve_model_core(model)
        Attention = model_core.layers[0].self_attn.__class__
        Model = model_core.__class__
    else:
        raise ValueError(f"Only supports llama, mistral and qwen2 models, not {model.__class__.__name__}.")

    first_attn = model_core.layers[0].self_attn
    dim, num_heads, num_heads_kv = _infer_attention_shape(first_attn, model_core)
    hf_rope = getattr(first_attn, "rotary_emb", None)
    if hf_rope is None:
        hf_rope = getattr(model_core, "rotary_emb", None)
    if isinstance(hf_rope, Qwen2RotaryEmbedding):
        rope_config = getattr(hf_rope, "config", None)
        rope_kwargs = getattr(hf_rope, "rope_kwargs", {}) or {}
        if rope_config is not None and hasattr(rope_config, "rope_theta"):
            base = rope_config.rope_theta
        elif "base" in rope_kwargs:
            base = rope_kwargs["base"]
        else:
            base = 10000.0
        distance_scale = 1.0
        if hasattr(hf_rope, "dim"):
            dim = hf_rope.dim
        elif getattr(hf_rope, "inv_freq", None) is not None:
            dim = int(hf_rope.inv_freq.shape[0] * 2)
    elif hf_rope is not None:
        base = hf_rope.config.rope_theta
        distance_scale = distance_scale if distance_scale is not None else 1.0
        partial_rotary_factor = hf_rope.config.partial_rotary_factor if hasattr(hf_rope.config, "partial_rotary_factor") else 1.0
        dim = int((hf_rope.config.hidden_size // hf_rope.config.num_attention_heads) * partial_rotary_factor)
    else:
        config = getattr(model_core, "config", None)
        if config is None or not hasattr(config, "rope_theta"):
            raise ValueError(f"Could not infer rotary embedding settings for {model.__class__.__name__}.")
        base = config.rope_theta
        distance_scale = distance_scale if distance_scale is not None else 1.0
    rope_device = next(model.parameters()).device
    rope = RotaryEmbeddingESM(
        dim,
        base,
        distance_scale,
        device=rope_device,
    )
    model_core.position_bias = rope

    def set_forward(m):
        if isinstance(m, Attention):
            inferred_dim_head, inferred_num_heads, inferred_num_heads_kv = _infer_attention_shape(m, model_core)
            if not hasattr(m, "head_dim"):
                m.head_dim = inferred_dim_head
            if not hasattr(m, "num_heads"):
                m.num_heads = inferred_num_heads
            if not hasattr(m, "num_key_value_heads"):
                m.num_key_value_heads = inferred_num_heads_kv
            m._old_forward = m.forward
            bound_forward = forward.__get__(m, Attention)

            def compat_forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask = None,
                position_ids = None,
                past_key_value = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position = None,
                position_embeddings = None,
                **inner_kwargs,
            ):
                del position_ids, cache_position, position_embeddings
                return bound_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=model_core.position_bias,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **inner_kwargs,
                )

            m.forward = compat_forward.__get__(m, Attention)

    model.apply(set_forward)

    model_core._old_forward = model_core.forward
    model_core.forward = model_forward.__get__(model_core, Model)

    return model
