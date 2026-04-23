import os
import sys
import time
from pathlib import Path

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
from lerobot.policies.factory import make_pre_post_processors

from configs import EXPERIMENTS, apply_canny

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(experiment_name: str):
    load_dotenv()

    exp = EXPERIMENTS[experiment_name]
    dataset_id = os.environ["DATASET_ID"]
    dataset_root = os.environ.get("DATASET_ROOT", None)
    policy_repo_id = f"{os.environ['POLICY_REPO_ID']}_{exp.name}"
    output_dir = Path(f"outputs/train/{exp.name}")

    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=dataset_root,
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
        dataset_meta = LeRobotDatasetMetadata(dataset_id, root=dataset_root)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset_meta = LeRobotDatasetMetadata(dataset_id, root=dataset_root)

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

    dataset = LeRobotDataset(dataset_id, root=dataset_root)
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
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)

    step = 0
    data_iter = iter(dataloader)

    while step < exp.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        t0 = time.perf_counter()
        batch = preprocessor(batch)

        if exp.use_canny:
            batch = apply_canny(batch, exp.canny_low, exp.canny_high)

        with accelerator.autocast():
            loss, _ = accelerator.unwrap_model(policy).forward(batch)

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), exp.grad_clip_norm)
        optimizer.step()

        step += 1
        step_time = time.perf_counter() - t0

        if accelerator.is_main_process and step % exp.log_freq == 0:
            print(
                f"[{exp.name}] step {step:>6} | loss {loss.item():.4f} | grad_norm {grad_norm.item():.3f} | {step_time * 1000:.0f}ms"
            )
            wandb.log(
                {
                    "loss": loss.item(),
                    "grad_norm": grad_norm.item(),
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
