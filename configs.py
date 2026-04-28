from dataclasses import dataclass
import cv2
import numpy as np
import torch


@dataclass
class ExperimentConfig:
    name: str
    vision_backbone: str
    pretrained_backbone_weights: str
    use_vae: bool
    use_canny: bool
    canny_low: int = 50
    canny_high: int = 150
    steps: int = 100_000
    batch_size: int = 8
    lr: float = 1e-4
    grad_clip_norm: float = 1.0
    log_freq: int = 100
    save_freq: int = 10_000


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "baseline": ExperimentConfig(
        name="baseline",
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=True,
        use_canny=False,
    ),
    "mobilenetv3_small": ExperimentConfig(
        name="mobilenetv3_small",
        vision_backbone="mobilenetv3_small_100",
        pretrained_backbone_weights="timm:mobilenetv3_small_100.lamb_in1k",
        use_vae=True,
        use_canny=False,
    ),
    "efficientnet_b0": ExperimentConfig(
        name="efficientnet_b0",
        vision_backbone="efficientnet_b0",
        pretrained_backbone_weights="timm:efficientnet_b0.ra_in1k",
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
    "no_vae": ExperimentConfig(
        name="no_vae",
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=False,
        use_canny=False,
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
            edge_tensor = torch.from_numpy(edges).to(device=imgs.device, dtype=imgs.dtype) / 255.0
            out[i] = edge_tensor.unsqueeze(0).repeat(3, 1, 1)
        batch[key] = out
    return batch
