from app.models.schemas import PredictionInput,PredictionOutput
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException, Request
from fastapi.responses import JSONResponse
import joblib
import uuid

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = joblib.load("ml/saved_model/model.joblib")
    app.state.model = model
    print("Model loaded successfully!")
    yield

class InvalidInputShape(Exception):
    pass

app = FastAPI(lifespan=lifespan)

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
def predict(data: PredictionInput):
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
        print("Prediction error:",e)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    request_id = str(uuid.uuid4())

    return {
        "prediction": int(prediction[0]),
        "confidence": confidence,
        "request_id": request_id
    }