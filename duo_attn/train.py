import os
import torch
from tqdm import tqdm
import json
import wandb
import matplotlib.pyplot as plt
from duo_attn.utils import (
    parse_args,
    get_tokenizer,
    visualize_pruned_attention_heads,
    full_attention_heads_to_list,
    save_full_attention_heads,
    seed_everything,
)
from duo_attn.data import (
    get_dataset,
    MultiplePasskeyRetrievalDataset,
    get_supervised_dataloader,
)
from duo_attn.loader import create_video_qa_dataloader
from duo_attn.patch import (
    enable_duo_attention_training,
    get_full_attention_heads,
    set_full_attention_heads,
    map_full_attention_heads,
    load_full_attention_heads,
)

from duo_attn.loss import l1_loss


import torch.distributed as dist

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
)

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer


def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def apply_fsdp(
    model: torch.nn.Module,
    mesh,
    mp_policy,
    modules_to_shard,
    shard_root: bool = True,
):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    fsdp_config = {"mp_policy": mp_policy, "mesh": mesh, "reshard_after_forward": True}

    for module in model.modules():
        if any([isinstance(module, m) for m in modules_to_shard]):
            fully_shard(module, **fsdp_config)

    if shard_root:
        fully_shard(model, **fsdp_config)


def _materialize_full_tensor(x: torch.Tensor) -> torch.Tensor:
    if hasattr(x, "full_tensor"):
        try:
            return x.full_tensor()
        except RuntimeError as exc:
            if (
                "allgather_into_tensor_coalesced" not in str(exc)
                or not dist.is_available()
                or not dist.is_initialized()
                or not hasattr(x, "to_local")
            ):
                raise

            # Fallback for NCCL builds that do not support the coalesced DTensor all-gather.
            local = x.to_local().contiguous()
            gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, local)
            return torch.cat(gathered, dim=0)
    return x


def move_batch_to_device(
    batch, device: torch.device, model_dtype: torch.dtype
):
    moved = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue

        if torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=model_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def build_llava_video_inputs_embeds(
    llava_model: LlavaOnevisionForConditionalGeneration,
    batch,
):
    input_embed_layer = llava_model.get_input_embeddings()
    input_ids = batch["input_ids"].to(input_embed_layer.weight.device)
    inputs_embeds = input_embed_layer(input_ids)

    if batch.get("pixel_values_videos", None) is not None:
        vision_dtype = next(llava_model.vision_tower.parameters()).dtype
        vision_device = next(llava_model.vision_tower.parameters()).device
        pixel_values_videos = batch["pixel_values_videos"].to(
            device=vision_device,
            dtype=vision_dtype,
        )
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        pixel_values_videos = pixel_values_videos.view(
            batch_size * frames, channels, height, width
        )

        vision_feature_layer = llava_model.config.vision_feature_layer
        vision_feature_select_strategy = llava_model.config.vision_feature_select_strategy

        # Avoid materializing all hidden states when we only need the final layer.
        if vision_feature_layer == -1:
            video_features = llava_model.vision_tower(
                pixel_values_videos,
                output_hidden_states=False,
            )
            selected_video_feature = video_features.last_hidden_state
        else:
            video_features = llava_model.vision_tower(
                pixel_values_videos,
                output_hidden_states=True,
            )
            selected_video_feature = video_features.hidden_states[vision_feature_layer]

        if vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature

        video_features = llava_model.multi_modal_projector(selected_video_feature)
        video_features = llava_model.apply_pooling(video_features)
        video_features = video_features.reshape(
            batch_size,
            frames * video_features.shape[1],
            -1,
        )

        image_newline = llava_model.image_newline[None, None, :].repeat(
            batch_size,
            1,
            1,
        ).to(video_features.device)
        video_features = torch.cat((video_features, image_newline), dim=1)
        video_features = video_features.flatten(0, 1)

        special_video_mask = (
            (input_ids == llava_model.config.video_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )

        video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)

    return inputs_embeds


def train(
    args,
    model,
    rank,
    world_size,
    train_dataloader,
    optimizer,
    scheduler,
    resume_step,
    llava_model_for_inputs=None,
):
    model.train()

    if rank == 0:
        pbar = tqdm(range(args.num_steps))

    local_rank = int(os.environ["LOCAL_RANK"])

    global_step = 0
    local_step = 0

    while True:
        if global_step >= args.num_steps:
            break
        for _, batch in enumerate(train_dataloader):
            if global_step <= resume_step:
                global_step += 1
                if rank == 0:
                    pbar.update(1)
                    pbar.set_description(
                        f"Skipping step {global_step} to resume to {resume_step}"
                    )
                continue

            @torch.no_grad()
            def clamp_(x, min_val, max_val):
                x.clamp_(min_val, max_val)

            map_full_attention_heads(model, func=lambda x: clamp_(x, 0, 1))

            batch = move_batch_to_device(
                batch,
                device=torch.device(f"cuda:{local_rank}"),
                model_dtype=next(model.parameters()).dtype,
            )

            if llava_model_for_inputs is None:
                model_inputs = torch.cat([batch["input_ids"], batch["input_ids"]], dim=0)
                model_kwargs = {
                    "input_ids": model_inputs,
                }
            else:
                with torch.no_grad():
                    inputs_embeds = build_llava_video_inputs_embeds(
                        llava_model_for_inputs,
                        batch,
                    )
                model_inputs = torch.cat([inputs_embeds, inputs_embeds], dim=0)
                model_kwargs = {
                    "inputs_embeds": model_inputs,
                }

            seq_len = model_inputs.shape[1]
            if seq_len % world_size != 0:
                raise ValueError(
                    f"Sequence length {seq_len} must be divisible by world_size {world_size}"
                )

            seq_parallel_chunk_size = seq_len // world_size
            seq_parallel_chunk_start = seq_parallel_chunk_size * rank
            seq_parallel_chunk_end = seq_parallel_chunk_start + seq_parallel_chunk_size

            local_batch_size = batch["labels"].shape[0]

            position_ids = torch.arange(
                seq_parallel_chunk_start,
                seq_parallel_chunk_end,
                device=model_inputs.device,
            ).unsqueeze(0)

            outputs = model(
                **{
                    k: v[:, seq_parallel_chunk_start:seq_parallel_chunk_end]
                    if k in {"input_ids", "inputs_embeds"}
                    else v
                    for k, v in model_kwargs.items()
                },
                position_ids=position_ids,
            )

            hidden_states = outputs[0]

            original_hidden_states = hidden_states[:local_batch_size]
            pruned_hidden_states = hidden_states[local_batch_size:]

            labels = batch["labels"][:, seq_parallel_chunk_start:seq_parallel_chunk_end]
            label_mask = labels != -100
            num_labels = label_mask.sum()
            global_num_labels = num_labels.clone().detach()
            dist.all_reduce(global_num_labels)

            # filter out label == IGNORE_INDEX (-100)
            if label_mask.any():
                original_hidden_states = original_hidden_states[label_mask].float()
                pruned_hidden_states = pruned_hidden_states[label_mask].float()
                distill_loss_num = (
                    (original_hidden_states - pruned_hidden_states)
                    .pow(2)
                    .mean(dim=-1)
                    .sum()
                )
            else:
                # Keep a zero-valued loss term connected to the graph so all ranks
                # execute identical backward collectives under sequence parallelism.
                distill_loss_num = hidden_states.float().sum() * 0.0

            distill_loss = (
                distill_loss_num * world_size / global_num_labels.clamp_min(1)
            )

            full_attention_heads_for_loss = get_full_attention_heads(model)
            full_attention_heads_for_loss = [
                _materialize_full_tensor(h).to(hidden_states.device)
                for h in full_attention_heads_for_loss
            ]

            reg_loss = l1_loss(torch.cat(full_attention_heads_for_loss).float())

            loss = distill_loss + args.reg_weight * reg_loss

            loss.backward()

            local_step = (local_step + 1) % args.gradient_accumulation_steps

            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(distill_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(reg_loss, op=dist.ReduceOp.AVG)

            if local_step != 0:
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            full_attention_heads_for_logging = get_full_attention_heads(model)
            full_attention_heads_for_logging = [
                _materialize_full_tensor(h).detach().clone()
                for h in full_attention_heads_for_logging
            ]
            if rank == 0:
                full_attention_heads_list = full_attention_heads_to_list(
                    full_attention_heads_for_logging
                )

                if not args.disable_wandb:
                    fig = visualize_pruned_attention_heads(full_attention_heads_list)

                    wandb.log(
                        {
                            "distill_loss": distill_loss.item(),
                            "reg_loss": reg_loss.item(),
                            "attn_heads": fig,
                            "step": global_step,
                            "sample_len": seq_len,
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

                    plt.close(fig)

                pbar.set_description(
                    f"Len={seq_len}/{global_num_labels.item()}|Dloss={distill_loss.item():.3f}|Rloss={reg_loss.item():.3f}|LR={optimizer.param_groups[0]['lr']:.2e}"
                )
                pbar.update(1)

            if args.output_dir is not None and global_step % args.save_steps == 0:
                if rank == 0:
                    save_full_attention_heads(
                        full_attention_heads_list,
                        os.path.join(
                            args.output_dir,
                            f"full_attention_heads_step={global_step}.tsv",
                        ),
                    )
                    os.system(f"rm -f {args.output_dir}/full_attention_heads_latest.tsv")
                    os.system(
                        f"cp {args.output_dir}/full_attention_heads_step={global_step}.tsv {args.output_dir}/full_attention_heads_latest.tsv"
                    )

                # save scheduler and optimizer state
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                    },
                    os.path.join(
                        args.output_dir,
                        f"optimizer_scheduler_state-step={global_step}-rank={rank}.pt",
                    ),
                )

                # copy the full_attention_heads and optimizer_scheduler_state to the latest state, replacing the old one
                # remove the previous latest state
                os.system(
                    f"rm -f {args.output_dir}/optimizer_scheduler_state_latest-rank={rank}.pt"
                )
                os.system(
                    f"cp {args.output_dir}/optimizer_scheduler_state-step={global_step}-rank={rank}.pt {args.output_dir}/optimizer_scheduler_state_latest-rank={rank}.pt"
                )

            if global_step >= args.num_steps:
                break

    if rank == 0:
        pbar.close()


def main(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if args.config_name is not None:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

    if args.rope_theta is not None:
        if hasattr(config, "rope_theta"):
            print(f"Setting rope_theta from {config.rope_theta} to {args.rope_theta}")
            config.rope_theta = args.rope_theta
        elif hasattr(config, "text_config") and hasattr(config.text_config, "rope_theta"):
            print(
                "Setting rope_theta for text_config "
                f"from {config.text_config.rope_theta} to {args.rope_theta}"
            )
            config.text_config.rope_theta = args.rope_theta

    is_llava_onevision = "llava_onevision" in config.model_type
    processor = None

    if is_llava_onevision:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            trust_remote_code=True,
        )
        tokenizer = processor.tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        tokenizer = get_tokenizer(args.model_name)

    enable_duo_attention_training(
        model,
        args.sink_size,
        args.recent_size,
        args.max_length,
        initial_value=args.initial_value,
        enable_ulysses_attention=True,
        streaming_attn_implementation=args.streaming_attn_implementation,
    )

    if is_llava_onevision:
        llava_model_for_inputs = model
        llava_model_for_inputs.eval()
        train_model = model.language_model.model
    else:
        llava_model_for_inputs = None
        train_model = model.model

    for param in train_model.parameters():
        param.requires_grad = False

    num_attn_heads = 0
    for name, param in train_model.named_parameters():
        if "full_attention_heads" in name:
            param.requires_grad = True
            num_attn_heads += param.numel()

    setup()

    torch.cuda.set_device(local_rank)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )

    if is_llava_onevision:
        llava_model_for_inputs.to(f"cuda:{local_rank}")

    apply_activation_checkpointing(train_model)

    mesh = DeviceMesh(device_type="cuda", mesh=[i for i in range(world_size)])

    # Llava path keeps root unsharded so multimodal embedding construction stays stable.
    shard_root = False if is_llava_onevision else True
    if args.disable_fsdp_root_shard:
        shard_root = False

    apply_fsdp(
        train_model,
        mesh,
        mp_policy,
        modules_to_shard={LlamaDecoderLayer, MistralDecoderLayer, Qwen2DecoderLayer},
        shard_root=shard_root,
    )

    if rank == 0:
        print(train_model)
        print(f"Total trainable full_attention_heads params: {num_attn_heads}")
        for name, param in train_model.named_parameters():
            if param.requires_grad:
                print(
                    f"Trainable parameter: {name} with shape {param.shape}, dtype {param.dtype}, device {param.device}"
                )

    if args.dataset_format == "multiple_passkey":
        haystack_dataset = get_dataset(args.dataset_name, split=args.split)
        train_dataset = MultiplePasskeyRetrievalDataset(
            haystack_dataset,
            tokenizer,
            max_length=args.max_length,
            min_depth_ratio=args.min_needle_depth_ratio,
            max_depth_ratio=args.max_needle_depth_ratio,
            context_length_min=args.context_length_min,
            context_length_max=args.context_length_max,
            context_lengths_num_intervals=args.context_lengths_num_intervals,
            depth_ratio_num_intervals=args.depth_ratio_num_intervals,
            num_passkeys=args.num_passkeys,
        )
        train_dataloader = get_supervised_dataloader(
            train_dataset,
            tokenizer,
            args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
    elif args.dataset_format == "video_qa":
        if not is_llava_onevision:
            raise ValueError(
                "dataset_format=video_qa currently requires a llava_onevision model."
            )

        train_dataloader = create_video_qa_dataloader(
            video_root=args.video_root,
            annotation_path=args.annotation_path,
            processor=processor,
            model_id=args.model_name,
            num_frames=args.num_frames,
            max_length=args.max_length,
            use_chat_template=not args.disable_video_chat_template,
            answer_prefix=args.video_answer_prefix,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            pad_to_multiple_of=world_size,
        )
    else:
        raise ValueError(f"Invalid dataset format: {args.dataset_format}")

    trainable_params = [p for p in train_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0)

    warmup_or_decay_window = max(args.num_steps // 5, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            1,
            max((step + 1) / warmup_or_decay_window, 0.1),
            max((args.num_steps - step) / warmup_or_decay_window, 0.1),
        ),
    )

    if rank == 0:
        experiment_config = vars(args)
        if not args.disable_wandb:
            wandb.init(project="DuoAttention", config=experiment_config)
            if args.exp_name is not None:
                wandb.run.name = args.exp_name

        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, "config.json"), "w") as f:
                json.dump(experiment_config, f)

    # if resume and link exists, load the latest state
    if args.resume and os.path.exists(
        os.path.join(
            args.output_dir, f"optimizer_scheduler_state_latest-rank={rank}.pt"
        )
    ):
        # load the latest state in the output_dir
        state = torch.load(
            os.path.join(
                args.output_dir, f"optimizer_scheduler_state_latest-rank={rank}.pt"
            )
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        full_attention_heads = load_full_attention_heads(
            args.output_dir, filename="full_attention_heads_latest.tsv"
        )
        set_full_attention_heads(train_model, full_attention_heads)
        resume_step = state["global_step"]
        print(f"Resuming from step {resume_step}")
    else:
        resume_step = -1

    train(
        args,
        train_model,
        rank,
        world_size,
        train_dataloader,
        optimizer,
        scheduler,
        resume_step,
        llava_model_for_inputs=llava_model_for_inputs,
    )

    full_attention_heads = get_full_attention_heads(train_model)
    full_attention_heads = [
        _materialize_full_tensor(h).detach().clone() for h in full_attention_heads
    ]

    if rank == 0:
        print("Training finished")
        if args.output_dir is not None:
            full_attention_heads_list = full_attention_heads_to_list(
                full_attention_heads
            )
            # save the full attention heads as tsv
            save_full_attention_heads(
                full_attention_heads_list,
                os.path.join(args.output_dir, "full_attention_heads.tsv"),
            )
    dist.barrier()
    cleanup()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    main(args)
    print("Done")
