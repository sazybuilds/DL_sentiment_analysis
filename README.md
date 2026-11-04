# Reel Feel

### A local movie-review sentiment analyzer

Reel Feel is a small, locally hosted web app for trying a deep-learning sentiment classifier. Paste a review, and a fine-tuned BERT model predicts whether it is **positive** or **negative** and returns its confidence score.

The interface uses plain HTML, CSS, and JavaScript. FastAPI serves both the page and the prediction API, so no separate frontend server or JavaScript build step is needed.

> **Before the first prediction:** this repository does not include trained model weights. Train the model once to create `DL_sentiment_analysis/src/model_dir/transformer.pt`. The first training run downloads the IMDb dataset and BERT's pretrained weights.

## What the model does

1. The BERT tokenizer turns review text into token IDs.
2. The pretrained `bert-base-uncased` encoder reads the tokens in context.
3. A trained two-class output layer uses the `[CLS]` representation to choose negative (`0`) or positive (`1`).
4. Softmax turns the two scores into probabilities. The API returns the winning label and its probability.

The model was designed for IMDb-style movie reviews. Its confidence score is not a guarantee, and the model may be wrong on sarcasm, mixed opinions, or text outside that domain. For a fuller explanation, see [flexisaf_submit_2.md](./flexisaf_submit_2.md).

## Requirements

- Python 3.10 or newer
- A working internet connection for the first training run and the first model load
- Enough memory and disk space for PyTorch, the IMDb dataset, and BERT
- A CUDA-capable GPU is optional; the application also runs on CPU

Install dependencies from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation, use the interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyTorch installation can depend on your operating system and GPU. If the default `torch` install is not suitable for your machine, install the matching PyTorch build using the instructions on the official PyTorch website, then install the remaining packages from `requirements.txt`.

## Train the model

Run this command from the repository root:

```powershell
python -m DL_sentiment_analysis.src.scripts.training
```

The default resource-conscious run trains the classifier layer while keeping BERT frozen, uses up to 2,000 training reviews and 500 reviews for validation and testing, and limits reviews to 128 tokens for one epoch. On CUDA, training also uses mixed precision and pinned-memory loading. The full IMDb data is split before sampling, so the selected validation and test examples remain separate from the training examples.

The selected model weights are saved to:

```text
DL_sentiment_analysis/src/model_dir/transformer.pt
```

The first run downloads the IMDb data and pretrained weights. The model still runs on CPU if no CUDA GPU is available, but training will take longer. Keep the saved checkpoint in place when launching the app.

To adjust the resource/quality tradeoff, use options such as:

```powershell
# More training data and more evaluation data, still with frozen BERT
python -m DL_sentiment_analysis.src.scripts.training --max-train-samples 5000 --max-eval-samples 1000 --epochs 2

# Full split, full-length reviews, and full BERT fine-tuning (requires substantially more time and memory)
python -m DL_sentiment_analysis.src.scripts.training --max-train-samples 0 --max-eval-samples 0 --max-length 512 --epochs 3 --unfreeze-bert
```

Sample limits of `0` mean use the full corresponding split. Use a smaller `--batch-size` if a GPU runs out of memory; full BERT fine-tuning is much more demanding than the default frozen-encoder run.

## Run the application

From the repository root, start the FastAPI server:

```powershell
python -m uvicorn DL_sentiment_analysis.src.backend.main:app --reload
```

Open **http://127.0.0.1:8000** in your browser. The page and API are served by the same local process.

Useful endpoints:

| URL | Purpose |
| --- | --- |
| `http://127.0.0.1:8000/` | Review analyzer UI |
| `http://127.0.0.1:8000/docs` | Interactive API documentation |
| `http://127.0.0.1:8000/api/health` | API and checkpoint status |
| `http://127.0.0.1:8000/api/predict` | Sentiment prediction endpoint (`POST`) |

Model weights are loaded the first time a prediction is requested. If the checkpoint is missing, the UI and API still start; the health status explains that training is required, and a prediction request returns HTTP `503` with the same guidance.

### Example API request

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"A thoughtful story with brilliant performances. I loved it."}'
```

Example response:

```json
{
  "prediction": "Positive",
  "class_id": 1,
  "probability": 0.97
}
```

`probability` is the winning class's softmax score, between `0` and `1`. The model truncates inputs to BERT's 512-token limit, and the API accepts at most 10,000 characters per review.

## Project map

```text
.
├── DL_sentiment_analysis/
│   └── src/
│       ├── backend/       FastAPI app, routes, request/response schemas, prediction service
│       ├── frontend/      Vanilla HTML, CSS, and JavaScript served by FastAPI
│       ├── model_dir/     BERT classifier definition and generated checkpoint
│       ├── notebook_dir/  Neural bag-of-words and BERT experiments
│       └── scripts/       Reproducible model-training script
├── utils/                 Dataset tokenization and batch-padding helpers
├── codebase_audit_and_improvements.md
├── codebase_walkthrough.md
├── flexisaf_submit_2.md
├── requirements.txt
└── transformers_notebook_explained.md
```

For a detailed, beginner-friendly explanation of what each file and code path does, read [codebase_walkthrough.md](./codebase_walkthrough.md). The notebook-specific walkthrough is [transformers_notebook_explained.md](./transformers_notebook_explained.md), and the original audit is [codebase_audit_and_improvements.md](./codebase_audit_and_improvements.md).

## Dependencies and project notes

- `transformers` and `uvicorn` are required by the app but were missing from the original dependency list; they are now included.
- `torchtext` remains because the separate neural bag-of-words notebook uses its tokenizer, vocabulary, and pretrained GloVe vectors. The BERT app and training script do not need it.
- The application has no Streamlit, Node.js, or frontend build requirement.
- Training and inference use the same model definition and checkpoint path.
