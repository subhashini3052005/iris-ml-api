from app.models.schemas import PredictionInput,PredictionOutput
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException, Request
from fastapi.responses import JSONResponse
from app.logging_config import setup_logger
import joblib
import uuid
import time

logger = setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = joblib.load("ml/saved_model/model.joblib")
    app.state.model = model
    logger.info("Model loaded successfully!")
    yield

class InvalidInputShape(Exception):
    pass

app = FastAPI(lifespan=lifespan)

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
        content={"detail": "Invalid input shape"}
    )


@app.get("/")
def root():
    return {"message": "ML API is alive"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model")
    }


@app.post("/predict",response_model=PredictionOutput)
def predict(request: Request, data: PredictionInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    if len(features) !=1 or len(features[0]) !=4:
        raise InvalidInputShape()
    try:
         prediction = app.state.model.predict(features)
         probabilities = app.state.model.predict_proba(features)
         confidence = float(max(probabilities[0]))

    except Exception as e :
        logger.error("Prediction error:%s",e)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    request_id = request.state.request_id

    logger.info(
        "Prediction successful: prediction=%s | request_id=%s",
        int(prediction[0]),
        request_id
    )

    return {
        "prediction": int(prediction[0]),
        "confidence": confidence,
        "request_id": request_id
    }