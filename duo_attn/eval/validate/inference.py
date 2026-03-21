import argparse
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import LlavaOnevisionForConditionalGeneration

from duo_attn.patch import enable_duo_attention_eval


def build_llava_video_inputs_embeds(
    model: LlavaOnevisionForConditionalGeneration,
    model_inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    input_embed_layer = model.get_input_embeddings()
    input_ids = model_inputs["input_ids"].to(input_embed_layer.weight.device)
    inputs_embeds = input_embed_layer(input_ids)

    if model_inputs.get("pixel_values_videos", None) is None:
        return inputs_embeds

    vision_dtype = next(model.vision_tower.parameters()).dtype
    vision_device = next(model.vision_tower.parameters()).device
    pixel_values_videos = model_inputs["pixel_values_videos"].to(
        device=vision_device,
        dtype=vision_dtype,
    )
    batch_size, frames, channels, height, width = pixel_values_videos.shape
    pixel_values_videos = pixel_values_videos.view(
        batch_size * frames,
        channels,
        height,
        width,
    )

    vision_feature_layer = model.config.vision_feature_layer
    vision_feature_select_strategy = model.config.vision_feature_select_strategy

    if vision_feature_layer == -1:
        video_features = model.vision_tower(
            pixel_values_videos,
            output_hidden_states=False,
        )
        selected_video_feature = video_features.last_hidden_state
    else:
        video_features = model.vision_tower(
            pixel_values_videos,
            output_hidden_states=True,
        )
        selected_video_feature = video_features.hidden_states[vision_feature_layer]

    if vision_feature_select_strategy == "default":
        selected_video_feature = selected_video_feature[:, 1:]
    elif vision_feature_select_strategy == "full":
        selected_video_feature = selected_video_feature
    else:
        raise ValueError(
            "Unexpected vision_feature_select_strategy="
            f"{vision_feature_select_strategy}"
        )

    video_features = model.multi_modal_projector(selected_video_feature)
    video_features = model.apply_pooling(video_features)
    video_features = video_features.reshape(
        batch_size,
        frames * video_features.shape[1],
        -1,
    )
    image_newline = model.image_newline[None, None, :].repeat(batch_size, 1, 1).to(
        video_features.device
    )
    video_features = torch.cat((video_features, image_newline), dim=1)
    video_features = video_features.flatten(0, 1)

    special_video_mask = (
        (input_ids == model.config.video_token_index)
        .unsqueeze(-1)
        .expand_as(inputs_embeds)
        .to(inputs_embeds.device)
    )
    video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)
    return inputs_embeds


def prepare_model_inputs(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    model_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    model_inputs: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key == "labels":
            continue
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                model_inputs[key] = value.to(device=device, dtype=model_dtype)
            else:
                model_inputs[key] = value.to(device=device)
        else:
            model_inputs[key] = value
    return model_inputs


def build_generation_prompt_inputs(
    model_inputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    tokenizer,
    answer_prefix: str,
) -> Dict[str, torch.Tensor]:
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError(
            f"Generation eval expects labels shape [1, L], got {tuple(labels.shape)}."
        )

    answer_positions = torch.nonzero(labels[0] != -100, as_tuple=False)
    if answer_positions.numel() == 0:
        prompt_len = model_inputs["input_ids"].shape[1]
    else:
        prompt_len = int(answer_positions[0].item())

    prompt_input_ids = model_inputs["input_ids"][:, :prompt_len]
    answer_prefix = str(answer_prefix or "")
    if answer_prefix:
        prefix_ids = tokenizer(
            answer_prefix,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(prompt_input_ids.device)
        prompt_input_ids = torch.cat([prompt_input_ids, prefix_ids], dim=1)

    prompt_inputs = dict(model_inputs)
    prompt_inputs["input_ids"] = prompt_input_ids
    if "attention_mask" in model_inputs:
        prompt_inputs["attention_mask"] = torch.ones(
            prompt_input_ids.shape,
            dtype=model_inputs["attention_mask"].dtype,
            device=prompt_input_ids.device,
        )
    else:
        prompt_inputs["attention_mask"] = torch.ones(
            prompt_input_ids.shape,
            dtype=torch.bool,
            device=prompt_input_ids.device,
        )
    return prompt_inputs


def greedy_generate_from_prompt(
    model: LlavaOnevisionForConditionalGeneration,
    prompt_inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
    tokenizer,
) -> torch.Tensor:
    if max_new_tokens <= 0:
        return torch.empty(
            (prompt_inputs["input_ids"].shape[0], 0),
            dtype=torch.long,
            device=prompt_inputs["input_ids"].device,
        )

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
    }
    if model.generation_config.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = int(model.generation_config.pad_token_id)
    elif tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = int(tokenizer.pad_token_id)

    with torch.inference_mode():
        generated_full = model.generate(**prompt_inputs, **generation_kwargs)

    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    if generated_full.shape[1] <= prompt_len:
        return torch.empty(
            (generated_full.shape[0], 0),
            dtype=torch.long,
            device=generated_full.device,
        )
    return generated_full[:, prompt_len:]


def canonicalize_secret_word(text: str) -> str:
    return re.sub(r"[^a-z]", "", str(text).lower())


def extract_secret_word_candidate(text: str) -> str:
    text = str(text).strip().lower()
    if not text:
        return ""

    filler_tokens = {
        "the",
        "secret",
        "word",
        "is",
        "a",
        "an",
        "answer",
        "final",
        "option",
        "assistant",
        "user",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
    }

    tokens = re.findall(r"[a-z]+", text)
    meaningful = [tok for tok in tokens if tok not in filler_tokens]
    if not meaningful:
        return ""

    # Score only using the final meaningful token.
    return meaningful[-1]


def to_safe_text(value: Any) -> str:
    return str(value).encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def build_debug_prefix_text(
    dataset: Any,
    raw_sample: Dict[str, Any],
    prompt_inputs: Dict[str, torch.Tensor],
    tokenizer: Any,
    answer_prefix: str,
) -> str:
    question = str(raw_sample.get("question", "")).strip()
    if question and hasattr(dataset, "_build_prefix_text"):
        try:
            base_prefix = str(dataset._build_prefix_text(question))
            return to_safe_text(f"{base_prefix}{answer_prefix}")
        except Exception:
            pass

    fallback_prefix = tokenizer.decode(
        prompt_inputs["input_ids"][0].detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return to_safe_text(fallback_prefix)


def evaluate_model(
    model: LlavaOnevisionForConditionalGeneration,
    processor: Any,
    dataloader: Any,
    device: torch.device,
    dtype: torch.dtype,
    n_samples: Optional[int],
    max_new_tokens: int,
    answer_prefix: str,
) -> Tuple[float, int, int, List[Dict[str, Any]]]:
    total_samples = 0
    exact_match_samples = 0
    detailed_samples: List[Dict[str, Any]] = []

    dataset = dataloader.dataset
    raw_samples = getattr(dataset, "samples", None)

    for i, batch in enumerate(dataloader):
        if n_samples is not None and total_samples >= n_samples:
            break

        if n_samples is not None:
            remaining = n_samples - total_samples
            current_batch_size = batch["labels"].shape[0]
            if remaining < current_batch_size:
                batch = {
                    k: (v[:remaining] if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }

        labels = batch["labels"].to(device=device)
        model_inputs = prepare_model_inputs(batch, device, dtype)
        prompt_inputs = build_generation_prompt_inputs(
            model_inputs=model_inputs,
            labels=labels,
            tokenizer=processor.tokenizer,
            answer_prefix=answer_prefix,
        )
        generated_ids = greedy_generate_from_prompt(
            model=model,
            prompt_inputs=prompt_inputs,
            max_new_tokens=max_new_tokens,
            tokenizer=processor.tokenizer,
        )

        sample_index = i * dataloader.batch_size

        pred_text = to_safe_text(
            processor.tokenizer.decode(
            generated_ids[0].detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            ).strip()
        )

        raw_sample: Dict[str, Any] = {}
        gt_word = ""
        if raw_samples is not None and sample_index < len(raw_samples):
            maybe_raw_sample = raw_samples[sample_index]
            if isinstance(maybe_raw_sample, dict):
                raw_sample = maybe_raw_sample
                gt_word = to_safe_text(raw_sample.get("gt", "")).strip()

        if not gt_word:
            gt_token_ids = labels[0][labels[0] != -100]
            label_text = processor.tokenizer.decode(
                gt_token_ids.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            gt_word = to_safe_text(extract_secret_word_candidate(label_text))

        pred_word = extract_secret_word_candidate(pred_text)

        gt_norm = canonicalize_secret_word(gt_word)
        pred_norm = canonicalize_secret_word(pred_word)
        is_exact = bool(gt_norm) and bool(pred_norm) and (pred_norm == gt_norm)

        options_value = raw_sample.get("options", [])
        if isinstance(options_value, list):
            options_list = [to_safe_text(option) for option in options_value]
        else:
            options_list = []

        prefix_text = build_debug_prefix_text(
            dataset=dataset,
            raw_sample=raw_sample,
            prompt_inputs=prompt_inputs,
            tokenizer=processor.tokenizer,
            answer_prefix=answer_prefix,
        )

        detailed_samples.append(
            {
                "sample_index": int(sample_index),
                "model_output_text": pred_text,
                "target_label_text": to_safe_text(gt_word),
                "is_exact_match": bool(is_exact),
                "annotation_gt": to_safe_text(raw_sample.get("gt", "")).strip(),
                "question": to_safe_text(raw_sample.get("question", "")).strip(),
                "video": to_safe_text(raw_sample.get("video", "")).strip(),
                "options": options_list,
                "prefix_text": prefix_text,
            }
        )

        total_samples += 1
        exact_match_samples += int(is_exact)

    sample_accuracy = (
        float(exact_match_samples / total_samples) if total_samples > 0 else 0.0
    )
    return sample_accuracy, total_samples, exact_match_samples, detailed_samples


def evaluate_mask_with_fresh_model(
    args: argparse.Namespace,
    config: Any,
    processor: Any,
    dataloader: Any,
    device: torch.device,
    dtype: torch.dtype,
    mask: np.ndarray,
    sink_size: int,
    recent_size: int,
) -> Tuple[float, int, int, List[Dict[str, Any]]]:
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    enable_duo_attention_eval(
        model,
        mask,
        sink_size,
        recent_size,
    )

    try:
        accuracy, total, exact_match, detailed_samples = evaluate_model(
            model=model,
            processor=processor,
            dataloader=dataloader,
            device=device,
            dtype=dtype,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            answer_prefix=args.video_answer_prefix,
        )
    finally:
        del model
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                # If a prior CUDA kernel fault occurred, empty_cache can fail too.
                pass

    return accuracy, total, exact_match, detailed_samples
