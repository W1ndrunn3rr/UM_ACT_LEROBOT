from typing import Any

from numpy.typing import NDArray

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

from src.config import apply_image_transform_to_array

from .configuration_filtered_opencv import FilteredOpenCVCameraConfig


class FilteredOpenCVCamera(OpenCVCamera):
    def __init__(self, config: FilteredOpenCVCameraConfig):
        super().__init__(config)
        self.config = config

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return OpenCVCamera.find_cameras()

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        processed = super()._postprocess_image(image)
        return apply_image_transform_to_array(processed, self.config.filter_name)
