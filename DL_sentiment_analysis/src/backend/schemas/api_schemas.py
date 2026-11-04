#Using pydantic to create the structure of data for application communication
from pydantic import BaseModel, Field



class HealthResponse(BaseModel):
    message: str
    status: str
    checkpoint_available: bool
    model_loaded: bool


class PredictionResponse(BaseModel):
    prediction: str
    class_id: int
    probability: float


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
