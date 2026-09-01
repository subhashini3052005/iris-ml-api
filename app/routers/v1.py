from fastapi import HTTPException, Request, APIRouter
from app.models.schemas import PredictionInput,PredictionOutput
from app.exceptions import InvalidInputShape
import logging

logger = logging.getLogger("ml.api")

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

@router.post("/predict",response_model=PredictionOutput)
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
         prediction = request.app.state.model.predict(features)
         probabilities = request.app.state.model.predict_proba(features)
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