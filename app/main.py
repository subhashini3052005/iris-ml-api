from app.models.schemas import PredictionInput
from contextlib import asynccontextmanager
from fastapi import FastAPI
import joblib

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = joblib.load("ml/saved_model/model.joblib")
    app.state.model = model
    print("Model loaded successfully!")
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "ML API is alive"}


@app.post("/predict")
def predict(data: PredictionInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = app.state.model.predict(features)

    return {"prediction": int(prediction[0])}