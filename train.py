import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

import torch
import torch.backends.cudnn
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import make_pre_post_processors

from configs import EXPERIMENTS, apply_canny

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(experiment_name: str):
    load_dotenv()

    exp = EXPERIMENTS[experiment_name]
    if not exp.vision_backbone.startswith("resnet"):
        raise ValueError(
            f"Experiment '{experiment_name}' uses unsupported ACT backbone "
            f"'{exp.vision_backbone}'. Use one of the ResNet-based experiments: "
            f"{[name for name, cfg in EXPERIMENTS.items() if cfg.vision_backbone.startswith('resnet')]}."
        )
    dataset_id = os.environ["DATASET_ID"]
    dataset_root = os.environ.get("DATASET_ROOT", None)
    dataset_local_dir = Path(dataset_root) / dataset_id if dataset_root else None
    policy_repo_id = f"{os.environ['POLICY_REPO_ID']}_{exp.name}"
    output_dir = Path(f"outputs/train/{exp.name}")

    accelerator = Accelerator(
        mixed_precision="bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=dataset_local_dir,
        )
        wandb.init(
            project="pick_and_lift",
            name=exp.name,
            config={
                "experiment": exp.name,
                "dataset": dataset_id,
                "steps": exp.steps,
                "batch_size": exp.batch_size * accelerator.num_processes,
                "lr": exp.lr,
                "vision_backbone": exp.vision_backbone,
                "use_vae": exp.use_vae,
                "use_canny": exp.use_canny,
            },
        )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        dataset_meta = LeRobotDatasetMetadata(dataset_id, root=dataset_local_dir)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset_meta = LeRobotDatasetMetadata(dataset_id, root=dataset_local_dir)

    features = dataset_to_policy_features(dataset_meta.features)
    output_features = {
        k: v for k, v in features.items() if v.type is FeatureType.ACTION
    }
    input_features = {k: v for k, v in features.items() if k not in output_features}

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        vision_backbone=exp.vision_backbone,
        pretrained_backbone_weights=exp.pretrained_backbone_weights or None,
        use_vae=exp.use_vae,
    )

    delta_timestamps = resolve_delta_timestamps(cfg, dataset_meta)
    dataset = LeRobotDataset(dataset_id, root=dataset_local_dir, delta_timestamps=delta_timestamps)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=exp.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    policy = ACTPolicy(cfg)
    policy.train()

    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=dataset_meta.stats
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=exp.lr, fused=True)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=exp.lr_plateau_factor,
        patience=exp.lr_plateau_patience,
        min_lr=exp.min_lr,
    )
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)

    step = 0
    data_iter = iter(dataloader)
    pbar = tqdm(total=exp.steps, desc=exp.name, disable=not accelerator.is_main_process)

    while step < exp.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        t0 = time.perf_counter()

        if exp.use_canny:
            batch = apply_canny(batch, exp.canny_low, exp.canny_high)

        batch = preprocessor(batch)

        with accelerator.autocast():
            loss, _ = policy(batch)

        if not torch.isfinite(loss):
            if accelerator.is_main_process:
                print(f"Non-finite loss at step {step + 1}: {loss.item()}. Stopping training.")
            break

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), exp.grad_clip_norm)

        if not torch.isfinite(grad_norm):
            if accelerator.is_main_process:
                print(f"Non-finite grad norm at step {step + 1}: {grad_norm.item()}. Skipping optimizer step.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if step < exp.warmup_steps:
            warmup_scale = (step + 1) / max(exp.warmup_steps, 1)
            for param_group in optimizer.param_groups:
                param_group["lr"] = exp.lr * warmup_scale

        optimizer.step()

        if step >= exp.warmup_steps:
            plateau_scheduler.step(loss.detach().float().item())

        step += 1
        step_time = time.perf_counter() - t0
        pbar.update(1)

        if accelerator.is_main_process and step % exp.log_freq == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grad_norm.item():.3f}", ms=f"{step_time*1000:.0f}")
            wandb.log(
                {
                    "loss": loss.item(),
                    "grad_norm": grad_norm.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "step_time_ms": step_time * 1000,
                },
                step=step,
            )

        if accelerator.is_main_process and (
            step % exp.save_freq == 0 or step == exp.steps
        ):
            ckpt = output_dir / f"step_{step:07d}"
            accelerator.unwrap_model(policy).save_pretrained(ckpt)
            preprocessor.save_pretrained(ckpt)
            postprocessor.save_pretrained(ckpt)
            print(f"Saved checkpoint to {ckpt}")

        accelerator.wait_for_everyone()

    pbar.close()

    if accelerator.is_main_process:
        accelerator.unwrap_model(policy).push_to_hub(policy_repo_id)
        preprocessor.push_to_hub(policy_repo_id)
        postprocessor.push_to_hub(policy_repo_id)
        print(f"Pushed to hub: {policy_repo_id}")
        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in EXPERIMENTS:
        print("Usage: python train.py <experiment>")
        print(f"Available: {list(EXPERIMENTS.keys())}")
        sys.exit(1)
    main(sys.argv[1])
