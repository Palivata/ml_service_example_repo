import cv2
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File, HTTPException, UploadFile

from src.containers.container import Container
from src.routes.routers import router
from src.services.service import Service


@router.post("/predict")
@inject
async def predict(
    file: UploadFile = File(...), service: Service = Depends(Provide[Container.service])
):
    content = await file.read()
    try:
        bounding_boxes, image = service.get_bounding_boxes(content)
    except cv2.error:
        raise HTTPException(status_code=415, detail="Please select image")

    if not bounding_boxes:
        return {"barcodes": []}
    texts = service.get_text(image, bounding_boxes)

    result = service.parse_result(texts, bounding_boxes)
    return {"barcodes": result}


