import numpy as np

from src.services.model import OCRModel, SegmentationModel
from src.services.configs import OCRModelConfig, SegmentationModelConfig

class FakeSegmentationModel(SegmentationModel):
    def __init__(self, config: dict):
        self._config = SegmentationModelConfig(**config)
        self.build()
        self._min_area_for_detection = self._config.min_area_for_detection

    def build(self):
        self._compiled_model = lambda x: [np.ones((self._config.model_input_height, self._config.model_input_width), dtype=np.uint8)]
        self._output = 0


class FakeOcrModel(OCRModel):
    def __init__(self, config: dict):
        self._config = OCRModelConfig(**config)
        self.build()
        self._character_list = self._config.character_list

    def build(self):
        self._compiled_model = lambda x: [np.ones((52, 1, 13), dtype=np.uint8)]
        self._output = 0