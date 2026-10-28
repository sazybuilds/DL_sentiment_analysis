from pydantic import BaseModel

class HealthResponse(BaseModel):
    message: str
    status: str


class PredictionResponse(BaseModel):
    prediction: str
    probability: float


class PredictionRequest(BaseModel):
    text: str

