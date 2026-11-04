# How the Deep-Learning Sentiment Model Works

This project teaches a computer to read a movie review and estimate whether the writer feels **positive** or **negative** about the film. The main model is a fine-tuned BERT language model with a small two-choice classifier attached to it.

## The simple picture

Think of the model as a reader and a decision-maker working together:

```text
Review written in ordinary words
              |
              v
Tokenizer turns words into BERT's numbered vocabulary
              |
              v
BERT reads the numbered words in context
              |
              v
The [CLS] summary goes to a two-choice classifier
              |
              v
Negative or Positive, together with a confidence score
```

The reader is **`bert-base-uncased`**. It is a pretrained Transformer encoder: before this project, it learned useful language patterns from a large text collection. The project fine-tunes that encoder on labeled movie reviews so that its learned language representations become useful for sentiment classification.

## 1. The examples: IMDb movie reviews

The BERT experiment uses the IMDb dataset supplied by Hugging Face's `datasets` library. It has two official parts:

- **25,000 training reviews** are split into training and validation portions. The script uses 75% of this part for training and 25% for validation.
- **25,000 test reviews** are kept aside for the final evaluation after training.

Every review has a label:

| Label | Meaning |
| --- | --- |
| `0` | Negative review |
| `1` | Positive review |

The labels are examples the model can learn from; they are not words found by the tokenizer.

## 2. Turning text into tokens

BERT does not receive a sentence as letters. Its tokenizer breaks the sentence into vocabulary pieces and looks up a number for each piece. This method is called **WordPiece tokenization**. Common words may remain whole; less common words can be split into smaller pieces.

For example, the tokenizer adds BERT's special beginning and ending tokens to a review. The first token is `[CLS]`, whose position is used by this classifier as a summary of the whole review. Token IDs are numbers, so the model can process them as tensors.

The tokenizer is configured for truncation. BERT can read at most 512 tokens for this model, so a longer review is shortened to fit. The browser allows up to 10,000 characters, but the model may not see every word in a very long review.

## 3. Batches and padding

Reviews have different lengths. A computer processes a small group of them at once, and tensors in one batch need matching shapes. The helper code pads shorter token sequences with BERT's padding token until they match the longest sequence in that batch.

Padding is not part of the review. The batch helper therefore also creates an **attention mask**:

- `1` means “this position contains a real token.”
- `0` means “this position is only padding.”

The mask tells BERT to ignore padding when it computes contextual representations. Without it, the model could treat empty padding positions as words.

## 4. BERT reads context

The model definition wraps Hugging Face's pretrained Transformer in a PyTorch module. For every token, BERT produces a vector containing information about that token in the context of the other tokens. The final hidden-state tensor has the shape:

```text
[number of reviews in the batch, tokens per review, hidden size]
```

For `bert-base-uncased`, the hidden size is 768. The model selects the vector at token position zero, the `[CLS]` position. It applies `tanh` to that vector and passes it through a learned linear layer with two outputs:

```text
[CLS] vector (768 numbers)
        |
      tanh
        |
linear classifier (2 scores)
        |
negative score, positive score
```

Those two values are **logits**: raw scores, not probabilities. During training, `CrossEntropyLoss` compares the logits with the correct label. It internally performs the appropriate normalization for the loss calculation.

## 5. How training teaches the model

The training script fine-tunes both the BERT encoder and the classifier head:

1. It sets random seeds to make the experiment more repeatable.
2. It downloads the IMDb train and test splits and tokenizes each review.
3. It splits the training portion into training and validation data.
4. It makes batches of eight reviews, padding them and building attention masks.
5. It sends each batch through the model and calculates classification loss.
6. It uses Adam with a learning rate of `1e-5` to adjust the trainable weights.
7. It repeats training for three epochs. An epoch is one pass through the training examples.
8. After each epoch, it evaluates on validation data. Whenever validation loss improves, it saves the model's **state dictionary** (the learned parameter values).
9. After training, it reloads the best saved weights and measures loss and accuracy on the held-out test split.

The checkpoint contains weights, not the tokenizer or the BERT architecture. Inference rebuilds the same architecture and loads those weights. The pretrained BERT files are downloaded or read from the local Hugging Face cache.

Fine-tuning all of BERT is computationally demanding. A GPU is helpful; CPU training is supported but can be slow.

## 6. How one prediction is made

When the user submits a review in the browser:

1. Vanilla JavaScript sends the text as JSON to `POST /api/predict`.
2. FastAPI checks that the input is non-empty and no longer than 10,000 characters.
3. The prediction service loads the tokenizer, BERT backbone, and trained checkpoint on the first request. It reuses them for subsequent requests.
4. The tokenizer produces `input_ids` and an `attention_mask`; the service truncates to 512 tokens.
5. PyTorch runs the model in evaluation mode and under `inference_mode`, so it does not calculate training gradients.
6. **Softmax** converts the two logits to probabilities that add up to one.
7. The service chooses the class with the larger probability and maps it back to a readable label.
8. FastAPI returns the label, class ID, and winning probability to the browser.
9. The page displays the result and a confidence bar.

The API response looks like this:

```json
{
  "prediction": "Positive",
  "class_id": 1,
  "probability": 0.97
}
```

The probability is the model's softmax score for its chosen class. It is **not a promise that the prediction is correct**, nor does this project calibrate it to real-world accuracy.

## 7. Why the app can start without a checkpoint

The trained checkpoint is generated by training and is not included in this repository. The web server therefore does not try to download or construct BERT just to start. Its health endpoint reports whether the checkpoint exists, and the service loads the model lazily when the first review is submitted.

If the checkpoint is absent, the page still opens and explains that training is needed. A prediction request returns HTTP `503` with instructions to run:

```powershell
python -m DL_sentiment_analysis.src.scripts.training
```

This is different from returning a guess from an untrained model: the application refuses to present untrained weights as a meaningful sentiment prediction.

## 8. What the model can and cannot tell you

The model learns patterns from labeled IMDb reviews. It can use context to interpret phrases that a simple word counter might misunderstand, but it does not reason exactly like a person. It may struggle with sarcasm, irony, very short text, mixed reviews, unusual slang, non-English text, or subjects unlike movie reviews.

The class labels and confidence are best treated as a model estimate for exploration, not as a human judgment or a guaranteed fact.

## Related reading

- [codebase_walkthrough.md](./codebase_walkthrough.md) explains the source files and how they fit together.
- [transformers_notebook_explained.md](./transformers_notebook_explained.md) follows the BERT notebook experiment cell by cell.
- [README.md](./README.md) contains installation, training, and local launch commands.
