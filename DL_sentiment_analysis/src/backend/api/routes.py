from fastapi import APIRouter, HTTPException #import router and exception

#import schema for the prediction endpoint
from DL_sentiment_analysis.src.backend.schemas.api_schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
)
#import error and predictor classes
from DL_sentiment_analysis.src.backend.services.predictor import (
    ModelUnavailableError,
    predictor,
)

#instantiate the API router
router = APIRouter()


#health endpoint
@router.get("/health", response_model=HealthResponse)
def health_check():
    checkpoint_available = predictor.checkpoint_available #call predictor class method
    if checkpoint_available:
        message = "The API is running. The model will load on the first prediction."
    else:
        message = (
            "The API is running, but no trained checkpoint was found. "
            "Train the model before requesting a prediction."
        )

    return {
        "status": "ok",
        "message": message,
        "checkpoint_available": checkpoint_available,
        "model_loaded": predictor.is_loaded,
    }

#predict endpoint
@router.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    if not request.text.strip(): #No request sent 
        raise HTTPException(
            status_code=422,
            detail="Enter a review containing at least one non-space character.",
        )

    try:
        return PredictionResponse(**predictor.predict(request.text))
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
