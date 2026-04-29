from dataclasses import dataclass
from pathlib import Path

from lerobot.cameras.configs import ColorMode, Cv2Backends, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from src.config import EXPERIMENTS


@OpenCVCameraConfig.register_subclass("filtered_opencv")
@dataclass
class FilteredOpenCVCameraConfig(OpenCVCameraConfig):
    index_or_path: int | Path
    filter_name: str = "none"
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    fourcc: str | None = None
    backend: Cv2Backends = Cv2Backends.ANY

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.filter_name != "none" and self.filter_name not in EXPERIMENTS:
            raise ValueError(
                f"Unknown filter_name '{self.filter_name}'. Available: ['none', {', '.join(sorted(EXPERIMENTS))}]"
            )
        if self.color_mode != ColorMode.RGB:
            raise ValueError("FilteredOpenCVCameraConfig requires color_mode='rgb'.")
