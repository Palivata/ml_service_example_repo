from typing import Union, List, Tuple

import cv2
import numpy as np
from openvino import Core

from src.services.configs import OCRModelConfig, SegmentationModelConfig
class SegmentationModel:
    def __init__(self, config: dict) -> None:
        self._config = SegmentationModelConfig(**config)
        self.build()
        self._min_area_for_detection = self._config.min_area_for_detection

    def build(self) -> None:
        """
        Build the model
        :return:
        """
        ie = Core()
        model = ie.read_model(self._config.model_path)
        self._compiled_model = ie.compile_model(model=model, device_name="CPU")
        self._output = self._compiled_model.output(0)


    def preprocess_image(self, image: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the image for model input
        :param image: bytes of image
        :return: preprocessed image
        """
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        image = cv2.resize(
            image,
            (self._config.model_input_height, self._config.model_input_width),
            cv2.INTER_CUBIC,
        ).astype(np.float32)
        image /= 255.0
        image -= np.array([0.485, 0.456, 0.406])
        image /= np.array([0.229, 0.224, 0.225])
        image = np.transpose(image, (2, 0, 1))[np.newaxis, ...]
        return image, original_image

    def predict(self, image: bytes) -> Union[List[List[int]], np.ndarray]:
        """
        Predict the classes of the image
        :param image: bytes of image
        :return: boxes and original image
        """
        img, original_image = self.preprocess_image(image)

        mask = self._compiled_model([img])[self._output]
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), cv2.INTER_CUBIC)
        mask = (mask > self._config.threshold).astype(int)

        contours, _ = cv2.findContours(
            mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area >= self._min_area_for_detection:  # Filter out small contours
                bounding_boxes.append((x, y, x + w, y + h))

        return bounding_boxes, original_image


class OCRModel:
    def __init__(self, config: dict) -> None:
        self._config = OCRModelConfig(**config)

        self.build()
        self._character_list = self._config.character_list
    def build(self) -> None:
        """
        Build the model
        :return:
        """
        ie = Core()
        model = ie.read_model(self._config.model_path)
        self._compiled_model = ie.compile_model(model=model, device_name="CPU")
        self._output = self._compiled_model.output(0)


    def preprocess_image(self, image: np.ndarray, bounding_box: List[int]) -> np.ndarray:
        """
        Preprocess the image for model input
        :param image: bytes of image
        :return: preprocessed image
        """
        x_min, y_min, x_max, y_max = bounding_box
        current_image = image[y_min:y_max, x_min:x_max].astype(np.float32)
        h, w = current_image.shape[:2]
        tmp_w = min(int(w * (self._config.model_input_height / h)), self._config.model_input_width)
        current_image = cv2.resize(current_image, (tmp_w, self._config.model_input_height))

        dw = np.round(self._config.model_input_width - tmp_w).astype(int)
        if dw > 0:
            pad_left = dw // 2
            pad_right = dw - pad_left
            current_image = cv2.copyMakeBorder(
                current_image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
            )
        current_image /= 255.0
        current_image -= np.array([0.485, 0.456, 0.406])
        current_image /= np.array([0.229, 0.224, 0.225])
        current_image = np.transpose(current_image, (2, 0, 1))[np.newaxis, ...]
        return current_image
    def postprocess_model_output(self, output: np.ndarray) -> str:
        probs = np.exp(output) / np.sum(np.exp(output), axis=2, keepdims=True)
        predicted_indices = np.argmax(probs, axis=2)
        predicted_indices = predicted_indices.squeeze(1)
        decoded_sequence = []
        previous_char = None
        for index in predicted_indices:
            if index != 0:
                if index != previous_char:
                    decoded_sequence.append(self._character_list[int(index)])
            previous_char = index
        decoded_text = "".join(decoded_sequence)
        if len(decoded_text) < self._config.max_len:
            decoded_text = decoded_text.ljust(self._config.max_len, '0')
        elif len(decoded_text) > self._config.max_len:
            decoded_text = decoded_text[:self._config.max_len]
        return decoded_text

    def predict(self, image: np.ndarray, bounding_boxes: list) -> List[str]:
        """
        Predict the text on the image
        :param bounding_boxes: list of bounding boxes
        :param image: bytes of image
        :return: str with predicted text
        """
        texts = []
        for bounding_box in bounding_boxes:
            img = self.preprocess_image(image, bounding_box)

            output = self._compiled_model([img])[self._output]
            decoded_text = self.postprocess_model_output(output)
            texts.append(decoded_text)
        return texts
