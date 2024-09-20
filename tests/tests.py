from mock import FakeSegmentationModel,FakeOcrModel
from conftest import test_config, test_image

from src.containers.container import Container


def test_predict_overrided_model_end_to_end(
    test_container: Container, test_image: bytes, test_config: dict, test_boxes: list, test_texts: list, test_result: list):
    with (test_container.ocr_model.override(FakeOcrModel(test_config["models"]["ocr_model"])),
        test_container.segmentation_model.override(FakeSegmentationModel(test_config["models"]["segmentation_model"])),):
        segmentation_model = test_container.segmentation_model()
        service = test_container.service()
        image, original_image = segmentation_model.preprocess_image(test_image)
        assert image.shape[2] == test_config["models"]["segmentation_model"]["model_input_width"]
        assert image.shape[3] == test_config["models"]["segmentation_model"]["model_input_height"]
        bounding_boxes, original_image = service.get_bounding_boxes(test_image)
        assert bounding_boxes == test_boxes
        texts = service.get_text(original_image, bounding_boxes)
        assert texts == test_texts
        result = service.parse_result(texts, bounding_boxes)
        assert result == test_result
