from pydantic import BaseModel, conlist
from typing import List

class OCRModelConfig(BaseModel):
    model_path: str
    model_input_height: int
    model_input_width: int
    character_list: conlist(str, min_items=1)  # To enforce at least one character in the list
    max_len: int

class SegmentationModelConfig(BaseModel):
    model_path: str
    model_input_height: int
    model_input_width: int
    min_area_for_detection: int
    threshold: float