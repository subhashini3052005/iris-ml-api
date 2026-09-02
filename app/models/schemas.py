from pydantic import BaseModel, Field
from typing import List

class PredictionInput(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)

class PredictionOutput(BaseModel):
    prediction: int
    confidence: float
    request_id: str

class PredictionBatchInput(BaseModel):
    inputs: List[PredictionInput]

class PredictionBatchOutput(BaseModel):
    predictions: List[PredictionOutput]