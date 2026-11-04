"""Train the IMDb sentiment classifier."""

import argparse
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import transformers

from utils.utility import get_data_loader, tokenize_and_numericalize_example

from DL_sentiment_analysis.src.model_dir.model_definition import Transformer


CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1] / "model_dir" / "transformer.pt"
)
SEED = 1234
TRANSFORMER_NAME = "bert-base-uncased"
IMDB_DATA_FILES = {
    "train": "plain_text/train-00000-of-00001.parquet",
    "test": "plain_text/test-00000-of-00001.parquet",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs (default: 1).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=2000,
        help="Training examples to use; 0 uses the full training split (default: 2000).",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=500,
        help="Examples per validation/test split; 0 uses each full split (default: 500).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum tokenized review length (default: 128).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Examples per batch (default: 8).",
    )
    parser.add_argument(
        "--unfreeze-bert",
        action="store_true",
        help="Fine-tune all BERT layers instead of training only the classifier.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.max_train_samples < 0 or args.max_eval_samples < 0:
        parser.error("sample limits must be 0 (full split) or greater")
    if not 1 <= args.max_length <= 512:
        parser.error("--max-length must be between 1 and BERT's 512-token limit")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def limit_dataset(dataset, max_samples, seed):
    if max_samples == 0 or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def get_accuracy(prediction, label):
    predicted_classes = prediction.argmax(dim=-1)
    return predicted_classes.eq(label).float().mean()


def train(data_loader, model, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    use_amp = device.type == "cuda"

    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device, non_blocking=use_amp)
        attention_mask = batch["attention_mask"].to(device, non_blocking=use_amp)
        label = batch["label"].to(device, non_blocking=use_amp)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            prediction = model(ids, attention_mask=attention_mask)
            loss = criterion(prediction, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = label.size(0)
        total_loss += loss.item() * batch_size
        total_correct += prediction.argmax(dim=-1).eq(label).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def evaluate(data_loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device, non_blocking=use_amp)
            attention_mask = batch["attention_mask"].to(
                device, non_blocking=use_amp
            )
            label = batch["label"].to(device, non_blocking=use_amp)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                prediction = model(ids, attention_mask=attention_mask)
                loss = criterion(prediction, label)

            batch_size = label.size(0)
            total_loss += loss.item() * batch_size
            total_correct += prediction.argmax(dim=-1).eq(label).sum().item()
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def main():
    args = parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    train_data = datasets.load_dataset(
        "stanfordnlp/imdb", data_files=IMDB_DATA_FILES, split="train"
    )
    test_data = datasets.load_dataset(
        "stanfordnlp/imdb", data_files=IMDB_DATA_FILES, split="test"
    )
    train_valid_data = train_data.train_test_split(test_size=0.25, seed=SEED)
    train_data = limit_dataset(
        train_valid_data["train"], args.max_train_samples, SEED
    )
    valid_data = limit_dataset(
        train_valid_data["test"], args.max_eval_samples, SEED + 1
    )
    test_data = limit_dataset(test_data, args.max_eval_samples, SEED + 2)

    tokenizer = transformers.AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
    tokenize_kwargs = {
        "tokenizer": tokenizer,
        "max_length": args.max_length,
    }
    train_data = train_data.map(
        tokenize_and_numericalize_example,
        batched=True,
        fn_kwargs=tokenize_kwargs,
    )
    valid_data = valid_data.map(
        tokenize_and_numericalize_example,
        batched=True,
        fn_kwargs=tokenize_kwargs,
    )
    test_data = test_data.map(
        tokenize_and_numericalize_example,
        batched=True,
        fn_kwargs=tokenize_kwargs,
    )

    pad_index = tokenizer.pad_token_id
    train_data = train_data.with_format(type="torch", columns=["ids", "label"])
    valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
    test_data = test_data.with_format(type="torch", columns=["ids", "label"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_data_loader = get_data_loader(
        train_data,
        args.batch_size,
        pad_index,
        shuffle=True,
        pin_memory=pin_memory,
    )
    valid_data_loader = get_data_loader(
        valid_data, args.batch_size, pad_index, pin_memory=pin_memory
    )
    test_data_loader = get_data_loader(
        test_data, args.batch_size, pad_index, pin_memory=pin_memory
    )

    transformer = transformers.AutoModel.from_pretrained(TRANSFORMER_NAME)
    model = Transformer(transformer, output_dim=2, freeze=not args.unfreeze_bert)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    learning_rate = 1e-5 if args.unfreeze_bert else 1e-3
    optimizer = optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Device: {device}")
    print(f"Training examples: {len(train_data):,}")
    print(f"Validation examples: {len(valid_data):,}")
    print(f"Test examples: {len(test_data):,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"BERT encoder frozen: {not args.unfreeze_bert}")
    print(f"Learning rate: {learning_rate:g}")

    best_valid_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss, train_acc = train(
            train_data_loader, model, criterion, optimizer, scaler, device
        )
        valid_loss, valid_acc = evaluate(
            valid_data_loader, model, criterion, device
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"epoch: {epoch + 1}/{args.epochs}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    )
    test_loss, test_acc = evaluate(test_data_loader, model, criterion, device)
    print("TEST LOSS: ", test_loss)
    print("TEST_ACC: ", test_acc)


if __name__ == "__main__":
    main()
