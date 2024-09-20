import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from src.containers.container import Container
from src.routes import routes
from src.routes.routers import router


def create_app() -> FastAPI:
    container = Container()
    cfg = OmegaConf.load("configs/config.yaml")
    container.config.from_dict(cfg)
    container.wire([routes])

    app = FastAPI()
    app.include_router(router, prefix="/main_service", tags=["main_service"])
    return app


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, port=5053, host="0.0.0.0")
