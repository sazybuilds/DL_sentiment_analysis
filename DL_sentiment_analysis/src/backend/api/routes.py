from fastapi import APIRouter


from backend.schemas.api_schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)
from backend.services.predictor import predict_sentiment

router = APIRouter()

@router.get("/", response_model=HealthResponse)
def health_check():
    return {
        "status": "okay",
        "message": "first endpoint is working"
    }


@router.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    text = request.text
    result = predict_sentiment(text)
    return PredictionResponse(**result)

