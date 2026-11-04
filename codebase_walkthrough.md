# The Codebase, Explained Step by Step

This guide is written for someone learning how the project works. It explains the job of each project file, how information moves through the code, and why the model has each major piece. It covers the implemented application as well as both notebooks.

For the short installation and launch instructions, start with [README.md](./README.md). For the model concepts without the full file tour, read [flexisaf_submit_2.md](./flexisaf_submit_2.md).

## First, the whole project in one picture

The codebase has two related but different jobs:

1. **Training:** show BERT many labeled movie reviews and save its best learned weights.
2. **Using the model:** accept one review in the browser, send it to Python, and show the model's prediction.

```text
TRAINING
IMDb reviews -> tokenizer -> padded batches + attention masks
             -> BERT + classifier -> loss and weight updates
             -> best checkpoint: model_dir/transformer.pt

LOCAL APP
Browser (HTML/CSS/JavaScript) -> FastAPI route -> predictor service
                                                   |
                            tokenizer + BERT + saved checkpoint
                                                   |
Browser <- JSON: label + class ID + probability <-+
```

Training does not happen every time the page opens. Training creates a checkpoint file once; the app loads that checkpoint to make predictions.

## Project map

```text
.
├── README.md
├── flexisaf_submit_2.md
├── codebase_walkthrough.md
├── codebase_audit_and_improvements.md
├── transformers_notebook_explained.md
├── requirements.txt
├── utils/
│   └── utility.py
└── DL_sentiment_analysis/
    └── src/
        ├── backend/
        │   ├── main.py
        │   ├── api/routes.py
        │   ├── schemas/api_schemas.py
        │   └── services/predictor.py
        ├── frontend/
        │   ├── app.py
        │   ├── index.html
        │   ├── styles.css
        │   └── app.js
        ├── model_dir/
        │   ├── __init__.py
        │   ├── model_definition.py
        │   └── transformer.pt             created by training; not committed
        ├── notebook_dir/
        │   ├── 1 - Neural Bag of Words.ipynb
        │   └── 4 - Transformers.ipynb
        └── scripts/
            └── training.py
```

The folders named `backend`, `api`, `schemas`, `services`, `frontend`, and `scripts` organize the source code by responsibility. Python can import these directories as namespace packages when commands are run from the repository root.

## A small glossary before reading the code

| Word | Plain-language meaning |
| --- | --- |
| **Review / example** | One piece of text that the model will learn from or predict. |
| **Label** | The known answer attached to a training example: `0` negative, `1` positive. |
| **Tokenizer** | A converter that turns text into the number IDs understood by BERT. |
| **Tensor** | A container of numbers arranged in one or more dimensions. PyTorch performs calculations on tensors. |
| **Batch** | Several examples processed together. This is more efficient than handling each training example separately. |
| **Padding** | Extra pad-token IDs added so every sequence in a batch has the same length. |
| **Attention mask** | A list that marks real tokens with `1` and padding with `0`. |
| **Logits** | The classifier's two raw scores, before they are changed into probabilities. |
| **Loss** | A number measuring how wrong the model's answers were during training. Training tries to make it smaller. |
| **Epoch** | One full pass through all training examples. |
| **Checkpoint / state dictionary** | A file containing the learned parameter values needed to rebuild the trained model. |
| **Inference** | Using an already trained model to make a prediction, without changing its weights. |

## The model files

### `DL_sentiment_analysis/src/model_dir/model_definition.py`

This file defines the model's shape using PyTorch:

1. `Transformer` inherits from `torch.nn.Module`, the base class for a PyTorch model.
2. Its constructor receives a pretrained BERT encoder, the number of output classes, and a `freeze` switch.
3. BERT's `hidden_size` tells the code how long each output vector is. For this BERT model it is 768.
4. `self.fc` is a learned linear layer that converts the 768-number `[CLS]` vector into two class scores.
5. If `freeze` is true, the code disables learning for BERT's parameters. The training script sets it to false, so it fine-tunes the encoder as well as the classifier.
6. `forward(ids, attention_mask)` is the route a batch takes through the model. It gives token IDs and their mask to BERT, reads `last_hidden_state`, selects the first token's vector, applies `tanh`, and returns two logits.

The attention mask is important in training: padded positions are not words, and BERT should not treat them as words. The model does not ask BERT to return attention matrices because this classifier does not use those matrices.

The output is two logits, not a string and not a probability. The predictor turns the winning logit into a human-readable class and a softmax probability.

### `DL_sentiment_analysis/src/model_dir/__init__.py`

This file is intentionally empty. Its presence marks `model_dir` as a Python package in environments that use regular packages. It contains no model logic.

## Preparing examples for training

### `utils/utility.py`

This shared helper file has three jobs:

- **`tokenize_and_numericalize_example`** takes one dataset record and uses the supplied tokenizer to convert its `text` field into token IDs. It enables truncation so a review cannot exceed the model's input length.
- **`get_collate_fn`** builds a function for turning a list of examples into one batch. It pads the ID sequences to the same length, creates an attention mask by checking which IDs are not the padding ID, and stacks the labels into a tensor.
- **`get_data_loader`** gives PyTorch a dataset, batch size, collate function, and optional shuffle setting. The training loader shuffles examples; validation and test loaders do not need to.

The helper returns a dictionary with `ids`, `attention_mask`, and `label`. The training code sends the first two to BERT and uses the label as the known answer for calculating loss.

## Training the model

### `DL_sentiment_analysis/src/scripts/training.py`

Run it from the repository root with:

```powershell
python -m DL_sentiment_analysis.src.scripts.training
```

The script proceeds in this order:

1. **Imports and reproducibility.** It imports the Python packages used for data, tensors, optimization, progress display, and Transformers. It sets NumPy and PyTorch random seeds to `1234` to make the experiment more repeatable.
2. **Data.** `datasets.load_dataset("imdb", split=["train", "test"])` obtains the 25,000 training and 25,000 test examples.
3. **Tokenizer.** `AutoTokenizer.from_pretrained("bert-base-uncased")` obtains the matching BERT vocabulary and tokenizer. The first use may download files from Hugging Face.
4. **Tokenization.** Dataset `.map(...)` applies the helper to each review and stores its token IDs.
5. **Validation split.** One quarter of the official training portion becomes validation data. The split uses the same fixed seed.
6. **Tensor format and batches.** The datasets expose `ids` and `label` as PyTorch tensors. Data loaders create batches of eight and dynamically pad each batch.
7. **Model.** `AutoModel.from_pretrained(...)` loads the BERT encoder; `Transformer(...)` adds the project classifier head. `freeze = False` means both are trainable.
8. **Training setup.** Cross-entropy compares two output logits against labels `0` and `1`; Adam adjusts model weights with learning rate `1e-5`. The selected device is CUDA if available, otherwise CPU.
9. **Training loop.** `train(...)` switches to training mode, calculates logits and loss, clears old gradients, computes new gradients with `backward()`, and lets the optimizer update the weights.
10. **Validation loop.** `evaluate(...)` switches to evaluation mode and disables gradient calculation. Validation measures whether the model is improving without updating weights.
11. **Checkpoint selection.** When validation loss improves, the script saves `model.state_dict()` to `DL_sentiment_analysis/src/model_dir/transformer.pt`. It then reloads that best checkpoint and evaluates the held-out test data.

`get_accuracy` chooses the largest of the two logits for each review and compares that class with the known label. The script reports training and validation loss/accuracy for each epoch and final test loss/accuracy.

This is a full BERT fine-tuning experiment and can take a long time on CPU. The IMDb dataset and initial pretrained BERT weights need to be downloaded on the first run. Training creates the checkpoint; the checkpoint is not present in the repository by default.

## Serving predictions

### `DL_sentiment_analysis/src/backend/services/predictor.py`

This is the bridge between the web API and the trained PyTorch model.

- `TRANSFORMER_NAME` and `LABELS` keep the model name and class ordering in one place. Label index `0` is negative; index `1` is positive.
- `CHECKPOINT_PATH` is calculated from this Python file's location, not the terminal's current folder. Training and prediction therefore use the same checkpoint.
- `SentimentPredictor` remembers the selected CPU/GPU device and starts with no BERT model loaded.
- `checkpoint_available` checks whether the trained checkpoint file exists. `is_loaded` reports whether model weights are already in memory.
- `_load_model` loads the tokenizer, BERT encoder, classifier, and checkpoint only when a prediction is requested. A lock prevents two simultaneous first requests from loading the large model twice. It moves the model to the selected device and switches it to evaluation mode.
- If the checkpoint is absent, `ModelUnavailableError` is raised with the training command. The service does not silently use random, untrained classifier weights.
- `predict` tokenizes and truncates a review to 512 tokens, moves its token IDs and attention mask to the same device as the model, and runs the model inside `torch.inference_mode()`. That mode avoids storing training gradients during prediction.
- `softmax` turns the two logits into probabilities. The class with the larger probability maps to `"Negative"` or `"Positive"`. The response contains that label, its numeric `class_id`, and its probability.

The model is loaded once and reused for later requests. Importing the API module alone does not download BERT or require the checkpoint to exist.

### `DL_sentiment_analysis/src/backend/schemas/api_schemas.py`

Pydantic models define and validate the shape of API data:

- `PredictionRequest` requires a string with 1 to 10,000 characters.
- `PredictionResponse` contains a readable `prediction`, integer `class_id`, and numeric `probability`.
- `HealthResponse` describes API status, whether a checkpoint exists, and whether model weights have already been loaded.

FastAPI uses these models to validate incoming JSON, document endpoints, and shape outgoing JSON.

### `DL_sentiment_analysis/src/backend/api/routes.py`

This file defines the API endpoints:

- `GET /api/health` reports whether the API is alive, whether the checkpoint file exists, and whether the model has been loaded. The API can be healthy even when training has not happened yet.
- `POST /api/predict` accepts `{"text": "..."}`, rejects text that contains only whitespace, and calls the predictor. If there are no trained weights, it returns an explicit HTTP `503` instead of a fake prediction.

### `DL_sentiment_analysis/src/backend/main.py`

This creates the FastAPI application. It registers the API router under `/api`, serves frontend assets under `/assets`, and returns the UI's `index.html` from `/`.

The browser and API share one origin, so the app does not need a separate frontend server or permissive cross-origin settings. FastAPI also provides `/docs` for interactive API documentation.

### `DL_sentiment_analysis/src/backend/api/__init__.py` and `schemas/__init__.py`

These are empty package marker files. They do not execute the API or contain schemas.

## The browser interface

All browser code is under `DL_sentiment_analysis/src/frontend/`. It uses no framework and needs no Node.js installation.

### `frontend/app.py`

This small Python file calculates the path to the frontend directory. The FastAPI entry point imports that path so it can serve the page and assets no matter which folder the server process is started from.

### `frontend/index.html`

This is the page's content and structure. It defines the review input, example buttons, character counter, status indicator, result area, confidence bar, and links to the API documentation. The `label` and ARIA attributes give assistive technologies useful descriptions.

The page does not itself run the model. It loads the CSS and JavaScript from `/assets`.

### `frontend/styles.css`

This file controls the visual design: colors, spacing, responsive layout, buttons, result card, confidence bar, and reduced-motion behavior. The CSS uses system fonts and does not fetch external assets.

### `frontend/app.js`

This is the browser's behavior, written in plain JavaScript:

1. On page load, it requests `/api/health` and updates the local model status.
2. Clicking an example fills the text box. The counter updates as text changes.
3. Submitting the form checks for blank text and sends JSON to `/api/predict`.
4. If the API reports an error, the page displays the error text instead of pretending there was a prediction.
5. On success, the page shows the class, confidence percentage, and matching progress bar.
6. Clearing the form removes the text and any old result.

The UI inserts model text with `textContent`, not HTML, so the review and error strings are displayed as text instead of being interpreted as page markup.

## The notebooks

The notebooks are learning and experimentation documents. They are separate from the FastAPI app and are not needed to launch it.

### `notebook_dir/1 - Neural Bag of Words.ipynb`

This notebook builds a simpler baseline called a **neural bag-of-words** model. It tokenizes text with `torchtext`, builds a vocabulary, represents words with embeddings (including GloVe vectors), and combines word information to classify sentiment.

It is useful for comparing a more traditional text model with BERT. The notebook uses `torchtext`, which is why that package remains in `requirements.txt`; the BERT server and training script do not import it.

### `notebook_dir/4 - Transformers.ipynb`

This notebook is the main experiment behind the deployed model. It walks through the IMDb data, BERT tokenizer, dynamic padding, classifier head, training and validation loops, checkpoint selection, test evaluation, and example predictions.

The shared model definition and script implement the same main design. The app adds practical serving behavior: lazy checkpoint loading, attention masks, bounded input length, device placement, evaluation mode, and an API response.

For a cell-by-cell explanation of the notebook, see [transformers_notebook_explained.md](./transformers_notebook_explained.md).

## Dependency list and documentation files

### `requirements.txt`

This lists the Python packages used by the app, training code, and notebooks. In particular:

- `torch` runs the model and training loops.
- `transformers` provides BERT and its tokenizer.
- `datasets` downloads and prepares IMDb.
- `fastapi` defines the HTTP API; `uvicorn` runs it.
- `torchtext` supports the separate bag-of-words notebook.
- `numpy`, `tqdm`, and `matplotlib` support experiment code and notebook plots.
- `pydantic` is used by FastAPI's request and response models.

### `codebase_audit_and_improvements.md`

This is the original audit, which records bugs found before repairs. Its opening status note distinguishes that historical diagnosis from the current app.

### `transformers_notebook_explained.md`

This is a longer, beginner-friendly walkthrough of the Transformer notebook.

### `flexisaf_submit_2.md`

This explains the sentiment model's data, architecture, training, inference, and limitations in one focused document.

## Run and inspect the app

From the repository root, see [README.md](./README.md) for virtual-environment, installation, training, and server commands. The app URLs are:

| URL | What you will find |
| --- | --- |
| `/` | Browser interface |
| `/api/health` | API and checkpoint status |
| `/api/predict` | JSON prediction endpoint |
| `/docs` | Interactive FastAPI documentation |

If the page loads but predictions are unavailable, check the status in the footer and run the training command. This is expected until the checkpoint has been created.
