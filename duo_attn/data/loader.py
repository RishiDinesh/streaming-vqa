from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .dynamic import DynamicSyntheticVideoQADataset
from .vnbench import VideoQADataset


@dataclass
class VideoQACollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("Tokenizer must provide pad_token_id or eos_token_id.")

        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 1:
            seq_len = input_ids.shape[1]
            remainder = seq_len % self.pad_to_multiple_of
            if remainder != 0:
                pad_len = self.pad_to_multiple_of - remainder
                pad_shape = (input_ids.shape[0], pad_len)
                input_pad = torch.full(
                    pad_shape,
                    fill_value=pad_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                label_pad = torch.full(
                    pad_shape,
                    fill_value=-100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                input_ids = torch.cat([input_ids, input_pad], dim=1)
                labels = torch.cat([labels, label_pad], dim=1)

        ret = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(pad_token_id),
        }

        for key in instances[0].keys():
            if key in ret:
                continue
            values = [instance[key] for instance in instances]
            if not all(torch.is_tensor(v) for v in values):
                continue

            if all(v.shape == values[0].shape for v in values):
                ret[key] = torch.stack(values)
                continue

            if values[0].ndim == 1:
                ret[key] = torch.nn.utils.rnn.pad_sequence(
                    values, batch_first=True, padding_value=0
                )
                continue

            raise ValueError(
                f"Cannot collate key '{key}' with shapes: "
                f"{[tuple(v.shape) for v in values]}"
            )

        return ret


def create_video_qa_dataloader(
    video_root: str,
    dataset_name: str,
    annotation_path: Optional[str] = None,
    processor: Optional[Any] = None,
    model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    num_frames: int = 8,
    max_length: int = 2048,
    use_chat_template: bool = True,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = False,
    drop_last: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    num_needles: int = 5,
    min_depth_ratio: float = 0.2,
    max_depth_ratio: float = 0.8,
    frame_idx: Optional[List[int]] = None,
) -> DataLoader:
    if dataset_name == "dynamic_synthetic":
        dataset = DynamicSyntheticVideoQADataset(
            video_root=video_root,
            processor=processor,
            model_id=model_id,
            num_frames=num_frames,
            max_length=max_length,
            use_chat_template=use_chat_template,
            num_needles=num_needles,
            min_depth_ratio=min_depth_ratio,
            max_depth_ratio=max_depth_ratio,
            frame_idx=frame_idx,
        )
    elif dataset_name == "vnbench":
        dataset = VideoQADataset(
            video_root=video_root,
            annotation_path=annotation_path,
            processor=processor,
            model_id=model_id,
            num_frames=num_frames,
            max_length=max_length,
            use_chat_template=use_chat_template
        )
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    collator = VideoQACollator(
        tokenizer=dataset.tokenizer,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collator,
    )
