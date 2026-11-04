#script containing the predictor class
from pathlib import Path
from threading import Lock
from typing import TypedDict

import torch
import transformers

from DL_sentiment_analysis.src.model_dir.model_definition import Transformer

TRANSFORMER_NAME = "bert-base-uncased"
LABELS = ("Negative", "Positive")
MAX_LENGTH = 512
CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2] / "model_dir" / "transformer.pt"
)


class PredictionResult(TypedDict):
    prediction: str
    class_id: int
    probability: float


class ModelUnavailableError(RuntimeError):
    """Raised when predictions are requested before trained weights exist."""

#The sentiment predictor class
class SentimentPredictor:
    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Transformer | None = None
        self._tokenizer = None
        self._load_lock = Lock()

    @property
    def checkpoint_available(self) -> bool: #check if model loaded weights are available
        return self.checkpoint_path.is_file()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> None:
        if self.is_loaded:
            return

        with self._load_lock:
            if self.is_loaded:
                return
            if not self.checkpoint_available:
                raise ModelUnavailableError(
                    "The trained model checkpoint is missing. From the repository "
                    "root, run `python -m DL_sentiment_analysis.src.scripts.training` "
                    "to train the model and create it."
                )

            tokenizer = transformers.AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
            backbone = transformers.AutoModel.from_pretrained(TRANSFORMER_NAME)
            model = Transformer(backbone, output_dim=len(LABELS), freeze=False)
            state_dict = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model

    #predict method of the predictor class
    def predict(self, text: str) -> PredictionResult:
        self._load_model()
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("The model failed to initialize.")

        encoded = self._tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.inference_mode():
            logits = self._model(input_ids, attention_mask=attention_mask).squeeze(0)
            probabilities = torch.softmax(logits, dim=-1)
            class_id = int(probabilities.argmax().item())

        return {
            "prediction": LABELS[class_id],
            "class_id": class_id,
            "probability": float(probabilities[class_id].item()),
        }


predictor = SentimentPredictor()
