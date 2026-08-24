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
def predict():
    features = [[5.1, 3.5, 1.4, 0.2]]

    prediction = app.state.model.predict(features)

    return {"prediction": int(prediction[0])}