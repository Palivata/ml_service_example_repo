import cv2
import numpy as np
import pytest

from src.containers.container import Container





@pytest.fixture
def test_image():
    img = np.random.uniform(size=(1080, 1920, 3))
    img = cv2.imencode(".jpg", img)[1].tobytes()
    return img


@pytest.fixture
def test_config():
    return {
        "models": {
            "segmentation_model": {"model_input_height": 224, "model_input_width": 225, "model_path": "", "min_area_for_detection":100,
                                   "threshold": 0.5},
            "ocr_model": {"model_input_height": 96, "model_input_width": 418, "model_path": "",
                          'character_list': ['blank', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                          'max_len': 13}
        },
    }


@pytest.fixture
def test_container(test_config):
    container = Container()

    container.config.from_dict(test_config)
    return container


@pytest.fixture
def test_boxes():
    return [(0, 0, 1920, 1080)]

@pytest.fixture
def test_texts():
    return ['0000000000000']

@pytest.fixture
def test_result():
    return [{'bbox': {'x_min': 0, 'y_min': 0, 'x_max': 1920, 'y_max': 1080}, 'value': '0000000000000'}]
