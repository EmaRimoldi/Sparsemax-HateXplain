from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import ExperimentConfig, load_config
from .data import HateXplainDataset, balanced_class_weights
from .model import SupervisedAttentionBert


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_datasets(
    config: ExperimentConfig,
    tokenizer: Any,
) -> tuple[HateXplainDataset, HateXplainDataset, HateXplainDataset]:
    common = {
        "dataset_path": config.dataset_path,
        "splits_path": config.splits_path,
        "tokenizer": tokenizer,
        "max_length": config.max_length,
        "target_normalization": config.target_normalization,
        "rationale_source": config.rationale_source,
        "rationale_aggregation": config.rationale_aggregation,
        "random_rationale_seed": config.random_rationale_seed,
        "normal_rationale_policy": config.normal_rationale_policy,
    }
    return (
        HateXplainDataset(split="train", **common),
        HateXplainDataset(split="val", **common),
        HateXplainDataset(split="test", **common),
    )


def make_loader(
    dataset: HateXplainDataset,
    config: ExperimentConfig,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key in {
            "input_ids",
            "attention_mask",
            "rationale_target",
            "rationale_weight",
            "label",
        }
    }


def evaluate(
    model: SupervisedAttentionBert,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    losses: list[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate", leave=False):
            tensors = move_batch(batch, device)
            output = model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                labels=tensors["label"],
                rationale_target=tensors["rationale_target"],
                rationale_weight=tensors["rationale_weight"],
            )
            if output.loss is None:
                raise RuntimeError("evaluation loss was not computed")
            losses.append(output.loss.item())
            labels.extend(tensors["label"].cpu().tolist())
            predictions.extend(output.logits.argmax(dim=-1).cpu().tolist())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(labels, predictions, average="macro")),
    }


def save_checkpoint(
    path: Path,
    model: SupervisedAttentionBert,
    config: ExperimentConfig,
    epoch: int,
    validation: dict[str, float],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "experiment": asdict(config),
            "epoch": epoch,
            "validation": validation,
        },
        path,
    )


def train(config: ExperimentConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = select_device()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    train_data, validation_data, test_data = make_datasets(config, tokenizer)
    class_weights = balanced_class_weights(train_data.label_ids)

    model = SupervisedAttentionBert.from_pretrained(config, class_weights).to(device)
    train_loader = make_loader(train_data, config, shuffle=True)
    validation_loader = make_loader(validation_data, config, shuffle=False)
    test_loader = make_loader(test_data, config, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    checkpoint_path = output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    best_validation_f1 = float("-inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{config.epochs}"):
            tensors = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                labels=tensors["label"],
                rationale_target=tensors["rationale_target"],
                rationale_weight=tensors["rationale_weight"],
            )
            if output.loss is None:
                raise RuntimeError("training loss was not computed")
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(output.loss.item())

        validation = evaluate(model, validation_loader, device)
        epoch_record = {
            "epoch": epoch,
            "training_loss": float(np.mean(epoch_losses)),
            "validation": validation,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True))

        if validation["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation["macro_f1"]
            save_checkpoint(checkpoint_path, model, config, epoch, validation)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    model.bert.config.save_pretrained(output_dir / "bert_config")

    summary = {
        "configuration": asdict(config),
        "device": str(device),
        "split_sizes": {
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_data),
        },
        "class_weights_from_training_split": class_weights.tolist(),
        "best_epoch": checkpoint["epoch"],
        "best_validation": checkpoint["validation"],
        "test": test_metrics,
        "history": history,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised-attention BERT variant")
    parser.add_argument("--config", required=True, help="Path to an experiment JSON file")
    args = parser.parse_args()
    summary = train(load_config(args.config))
    print(json.dumps(summary["test"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
