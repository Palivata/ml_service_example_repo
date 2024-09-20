from typing import List

import numpy as np

from src.services.model import SegmentationModel, OCRModel


class Service:
    def __init__(self, segmentation_model: SegmentationModel,
                 ocr_model: OCRModel) -> None:
        self._segmentation_model = segmentation_model
        self._ocr_model = ocr_model

    def get_bounding_boxes(self, content: bytes) -> dict:
        """
        :param content: image bytes
        :return: fname of stored image
        """
        bounding_boxes, original_image = self._segmentation_model.predict(content)

        return bounding_boxes, original_image

    def get_text(self, image: np.ndarray, bounding_boxes: List[str]) -> List[str]:
        """
        :param content: image bytes
        :return: fname of stored image
        """

        return self._ocr_model.predict(image, bounding_boxes)

    def parse_result(self, texts, boudning_boxes):
        result = []
        for text, bounding_box in zip(texts, boudning_boxes):
            x_min, y_min, x_max, y_max = bounding_box
            result.append({"bbox": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
                           "value": text})
        return result
