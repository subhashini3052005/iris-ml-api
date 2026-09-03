from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.logging_config import setup_logger
from app.routers.v1 import router as v1_router
from app.exceptions import InvalidInputShape
from app.config import settings
import joblib
import uuid
import time

logger = setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = joblib.load(settings.MODEL_PATH)
    app.state.model = model
    logger.info("Model loaded successfully!")
    yield

app = FastAPI(
    title = settings.API_TITLE,
    lifespan = lifespan
)

app.include_router(v1_router)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    logger.info(
        "Request: %s %s | request_id=%s | duration=%.4fs | status=%s",
        request.method,
        request.url.path,
        request_id,
        duration,
        response.status_code
    )

    return response

@app.exception_handler(InvalidInputShape)
async def invalid_input_shape_handler(request: Request, exc: InvalidInputShape):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.get("/")
def root():
    return {"message": "ML API is alive"}

