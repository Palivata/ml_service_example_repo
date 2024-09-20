from dependency_injector import containers, providers

from src.services.model import SegmentationModel, OCRModel
from src.services.service import Service


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    segmentation_model = providers.Singleton(SegmentationModel, config.models.segmentation_model)
    ocr_model = providers.Singleton(OCRModel, config.models.ocr_model)
    service = providers.Factory(Service, segmentation_model=segmentation_model, ocr_model=ocr_model)
