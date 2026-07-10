"""Command-line runner for reproducible date2date experiments."""

from __future__ import annotations

import argparse
import platform
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .attention_rnn import DecoderRNN_WithAttention
from .data import DEVICE, fixed_samples, vocab_size
from .experiment_log import (
    PROJECT_ROOT,
    RUNS_DIR,
    append_index,
    git_commit,
    load_or_create_fixed_test,
    make_training_samples,
    new_run_id,
    save_checkpoint,
    write_json,
)
from .report import update_readme
from .rnn import DecoderRNN, EncoderRNN
from .train import char_accuracy, eval_samples, generate, train_one_sample

MODEL_NAMES = ("vanilla_gru", "bahdanau")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(model_type: str, hidden_size: int) -> tuple[nn.Module, nn.Module]:
    encoder = EncoderRNN(vocab_size, hidden_size)
    if model_type == "vanilla_gru":
        decoder = DecoderRNN(vocab_size, hidden_size, vocab_size)
    elif model_type == "bahdanau":
        decoder = DecoderRNN_WithAttention(vocab_size, hidden_size, vocab_size)
    else:
        raise ValueError(f"Unsupported model: {model_type}")
    return encoder.to(DEVICE), decoder.to(DEVICE)


def qualitative_eval(
    samples: list[tuple[str, str]],
    encoder: nn.Module,
    decoder: nn.Module,
) -> list[dict[str, Any]]:
    return [
        {
            "input": input_str,
            "target": target_str,
            "prediction": prediction,
            "correct": prediction == target_str,
        }
        for input_str, target_str in samples
        for prediction in [generate(input_str, encoder, decoder)]
    ]


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    run_id = new_run_id()
    run_dir = RUNS_DIR / run_id
    test_samples = load_or_create_fixed_test(args.test_seed, args.test_size)
    train_samples = make_training_samples(
        args.train_size,
        args.seed,
        {input_str for input_str, _ in test_samples},
    )

    encoder, decoder = build_models(args.model, args.hidden_size)
    if hasattr(encoder, "dropout"):
        encoder.dropout.p = args.dropout
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    config = {
        "model_type": args.model,
        "hidden_size": args.hidden_size,
        "lr": args.lr,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "optimizer": "Adam",
        "seed": args.seed,
        "test_seed": args.test_seed,
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    write_json(run_dir / "config.json", config)

    started_at = time.perf_counter()
    epoch_losses = []
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        random.shuffle(train_samples)
        epoch_loss = sum(
            train_one_sample(
                input_str,
                target_str,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                criterion,
            )
            for input_str, target_str in train_samples
        )
        avg_loss = epoch_loss / len(train_samples)
        epoch_losses.append(avg_loss)
        print(f"epoch {epoch + 1}/{args.epochs} avg_loss={avg_loss:.6f}")

    train_correct, train_total, train_acc = eval_samples(
        train_samples, encoder, decoder
    )
    test_correct, test_total, test_acc = eval_samples(test_samples, encoder, decoder)
    test_char_acc = char_accuracy(test_samples, encoder, decoder)
    duration = time.perf_counter() - started_at
    parameter_count = sum(
        parameter.numel()
        for model in (encoder, decoder)
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    predictions = qualitative_eval(fixed_samples, encoder, decoder)

    checkpoint_path = run_dir / "checkpoint.pt"
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.json"
    metrics = {
        "final_loss": epoch_losses[-1],
        "epoch_losses": epoch_losses,
        "train_correct": train_correct,
        "train_total": train_total,
        "train_exact_match": train_acc,
        "test_correct": test_correct,
        "test_total": test_total,
        "test_exact_match": test_acc,
        "test_char_accuracy": test_char_acc,
        "parameter_count": parameter_count,
        "duration_seconds": duration,
    }
    write_json(metrics_path, metrics)
    write_json(predictions_path, predictions)
    save_checkpoint(checkpoint_path, encoder, decoder, config, metrics)

    record = {
        "run_id": run_id,
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        **config,
        **metrics,
        "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
        "notes": args.notes,
    }
    append_index(record)
    if not args.no_update_readme:
        update_readme()

    print(
        f"test_exact_match={test_acc:.4f} "
        f"test_char_accuracy={test_char_acc:.4f} duration={duration:.1f}s"
    )
    print(f"run artifacts: {run_dir.relative_to(PROJECT_ROOT)}")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行并记录 date2date 实验")
    parser.add_argument("--model", choices=MODEL_NAMES, default="bahdanau")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-seed", type=int, default=20260710)
    parser.add_argument("--notes", default="")
    parser.add_argument("--no-update-readme", action="store_true")
    args = parser.parse_args()
    for name in ("hidden_size", "train_size", "test_size", "epochs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than 0")
    if args.lr <= 0:
        parser.error("--lr must be greater than 0")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    return args


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
