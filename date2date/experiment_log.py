"""Experiment artifact and index persistence."""

from __future__ import annotations

import csv
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from .data import itos, make_sample, stoi, vocab_size

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
INDEX_PATH = EXPERIMENTS_DIR / "experiments.csv"

INDEX_FIELDS = [
    "run_id",
    "date",
    "git_commit",
    "model_type",
    "hidden_size",
    "lr",
    "train_size",
    "test_size",
    "epochs",
    "batch_size",
    "dropout",
    "optimizer",
    "seed",
    "test_seed",
    "parameter_count",
    "final_loss",
    "train_exact_match",
    "test_exact_match",
    "test_char_accuracy",
    "duration_seconds",
    "device",
    "checkpoint_path",
    "metrics_path",
    "notes",
]


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fixed_test_path(test_seed: int, test_size: int) -> Path:
    return EXPERIMENTS_DIR / f"fixed_test_seed{test_seed}_n{test_size}.json"


def _validate_fixed_test(
    payload: dict[str, Any], test_seed: int, test_size: int
) -> list[tuple[str, str]]:
    samples = [
        (item["input"], item["target"])
        for item in payload.get("samples", [])
    ]
    if payload.get("seed") != test_seed or payload.get("size") != test_size:
        raise ValueError("固定测试集的 seed 或 size 与文件名不一致")
    if len(samples) != test_size:
        raise ValueError("固定测试集的实际样本数与 size 不一致")
    if len({input_str for input_str, _ in samples}) != test_size:
        raise ValueError("固定测试集包含重复输入，请删除后重新生成")
    if any(
        len(input_str) != 10 or len(target_str) != 10
        for input_str, target_str in samples
    ):
        raise ValueError("固定测试集包含格式错误的样本")
    return samples


def load_or_create_fixed_test(test_seed: int, test_size: int) -> list[tuple[str, str]]:
    path = fixed_test_path(test_seed, test_size)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _validate_fixed_test(payload, test_seed, test_size)

    rng = random.Random(test_seed)
    samples: list[tuple[str, str]] = []
    seen: set[str] = set()
    while len(samples) < test_size:
        inputs, targets = make_sample(1, rng=rng)
        if inputs[0] in seen:
            continue
        seen.add(inputs[0])
        samples.append((inputs[0], targets[0]))
    payload = {
        "seed": test_seed,
        "size": test_size,
        "samples": [
            {"input": input_str, "target": target_str}
            for input_str, target_str in samples
        ],
    }
    write_json(path, payload)
    return samples


def make_training_samples(
    train_size: int,
    seed: int,
    excluded_inputs: set[str],
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    samples: list[tuple[str, str]] = []
    seen = set(excluded_inputs)
    while len(samples) < train_size:
        inputs, targets = make_sample(1, rng=rng)
        if inputs[0] in seen:
            continue
        seen.add(inputs[0])
        samples.append((inputs[0], targets[0]))
    return samples


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_checkpoint(
    path: Path,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "config": config,
            "metrics": metrics,
            "vocab_size": vocab_size,
            "stoi": stoi,
            "itos": itos,
        },
        path,
    )


def append_index(record: dict[str, Any]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = INDEX_PATH.exists()
    with INDEX_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=INDEX_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
