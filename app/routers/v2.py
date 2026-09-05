from fastapi import APIRouter, Request, HTTPException
from app.models.schemas import PredictionInput, PredictionV2Output
import logging

router= APIRouter(prefix="/api/v2")

@router.post("/predict", response_model=PredictionV2Output)
def predict_v2(request: Request, data: PredictionInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    try:
        prediction = request.app.state.model.predict(features)
        probabilities = request.app.state.model.predict_proba(features)

    except Exception as e:
        logging.error("V2 prediction error: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    request_id = request.state.request_id

    return {
        "prediction": int(prediction[0]),
        "probabilities": [
            float(probability)
            for probability in probabilities[0]
        ],
        "request_id": request_id
    }