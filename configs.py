from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
import torch


@dataclass
class ExperimentConfig:
    name: str
    vision_backbone: str
    pretrained_backbone_weights: Optional[str]
    use_vae: bool
    use_canny: bool
    canny_low: int = 50
    canny_high: int = 150
    steps: int = 50_000
    batch_size: int = 8
    lr: float = 5e-5
    grad_clip_norm: float = 0.5
    warmup_steps: int = 2_000
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 1_000
    min_lr: float = 1e-6
    log_freq: int = 100
    save_freq: int = 10_000


EXPERIMENTS: dict[str, ExperimentConfig] = {
    # "baseline": ExperimentConfig(
    #     name="baseline",
    #     vision_backbone="resnet18",
    #     pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
    #     use_vae=True,
    #     use_canny=False,
    # ),
    "no_vae": ExperimentConfig(
        name="no_vae",
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=False,
        use_canny=False,
    ),
    "resnet50_scratch": ExperimentConfig(
        name="resnet50_scratch",
        vision_backbone="resnet50",
        pretrained_backbone_weights=None,
        use_vae=True,
        use_canny=False,
    ),
    "resnet50_pretrained": ExperimentConfig(
        name="resnet50_pretrained",
        vision_backbone="resnet50",
        pretrained_backbone_weights="ResNet50_Weights.IMAGENET1K_V2",
        use_vae=True,
        use_canny=False,
    ),
    "canny": ExperimentConfig(
        name="canny",
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=True,
        use_canny=True,
        canny_low=50,
        canny_high=150,
    ),
}


def apply_canny(batch: dict, low: int, high: int) -> dict:
    for key in list(batch.keys()):
        if "image" not in key:
            continue
        imgs = batch[key]
        B, C, H, W = imgs.shape
        out = torch.zeros(B, 3, H, W, dtype=imgs.dtype, device=imgs.device)
        for i in range(B):
            img_np = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, low, high)
            edge_tensor = (
                torch.from_numpy(edges).to(device=imgs.device, dtype=imgs.dtype) / 255.0
            )
            out[i] = edge_tensor.unsqueeze(0).repeat(3, 1, 1)
        batch[key] = out
    return batch
