from fastapi import HTTPException, Request, APIRouter
from app.models.schemas import (
    PredictionInput,
    PredictionOutput,
    PredictionBatchInput,
    PredictionBatchOutput
    )
from app.exceptions import InvalidInputShape
from app.config import settings
import logging
import time
import json

logger = logging.getLogger("ml_api")

router = APIRouter(prefix="/api/v1")

# V2 plan:
# If /api/v2/predict is needed, I will create a separate v2 router
# with its own response schema. The v2 response can include extra
# fields without changing the existing v1 response contract.

@router.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": hasattr(request.app.state, "model")
    }

@router.get("/model-info")
def model_info():
    with open("ml/saved_model/model_info.json", "r") as f:
        metadata = json.load(f)
    return metadata

@router.post("/predict-batch",response_model=PredictionBatchOutput)
def predict_batch(request: Request, data: PredictionBatchInput):
    features = [
        [
        item.sepal_length,
        item.sepal_width,
        item.petal_length,
        item.petal_width
        ]
        for item in data.inputs
    ]

    if len(data.inputs)<1 or len(data.inputs)>settings.MAX_BATCH_SIZE:
        raise InvalidInputShape(
            f"Batch size must be between 1 and {settings.MAX_BATCH_SIZE}"
        )

    start_time = time.time()

    try:
         prediction = request.app.state.model.predict(features)
         probabilities = request.app.state.model.predict_proba(features)
         confidences = [max(probability) for probability in probabilities]

    except Exception as e :
        logger.error("Prediction error:%s",e)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    request_id = request.state.request_id
    duration = time.time() - start_time

    logger.info(
        "Batch prediction successful | batch_size=%s | duration=%.4fs | request_id=%s",
        len(data.inputs),
        duration,
        request_id
    )

    predictions = []

    for pred,conf in zip(prediction,confidences):
        predictions.append({
            "prediction": int(pred),
            "confidence": float(conf),
            "request_id": request_id
        })

    return{
        "predictions": predictions
    }
       
    