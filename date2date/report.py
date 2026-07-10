"""Render recorded experiment results into README."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .experiment_log import INDEX_PATH, PROJECT_ROOT

START_MARKER = "<!-- EXPERIMENT_RESULTS_START -->"
END_MARKER = "<!-- EXPERIMENT_RESULTS_END -->"
README_PATH = PROJECT_ROOT / "Readme.md"


def _percent(value: str) -> str:
    return f"{float(value) * 100:.2f}%"


def _best_runs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        model = row["model_type"]
        score = (
            float(row["test_exact_match"]),
            float(row["test_char_accuracy"]),
            row["date"],
        )
        current = best.get(model)
        if current is None:
            best[model] = row
            continue
        current_score = (
            float(current["test_exact_match"]),
            float(current["test_char_accuracy"]),
            current["date"],
        )
        if score > current_score:
            best[model] = row
    return sorted(best.values(), key=lambda row: row["model_type"])


def render_results(index_path: Path = INDEX_PATH) -> str:
    if not index_path.exists():
        return "_尚未记录自动化实验。_"

    with index_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return "_尚未记录自动化实验。_"

    lines = [
        "| 模型 | 测试集 | Exact Match | 字符准确率 | 参数量 | 训练耗时 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in _best_runs(rows):
        lines.append(
            "| {model} | {test_size} | {exact} | {char} | {params:,} | {duration:.1f}s |".format(
                model=row["model_type"],
                test_size=row["test_size"],
                exact=_percent(row["test_exact_match"]),
                char=_percent(row["test_char_accuracy"]),
                params=int(row["parameter_count"]),
                duration=float(row["duration_seconds"]),
            )
        )
    try:
        display_path = index_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = index_path
    lines.extend(
        [
            "",
            f"> 结果由 `{display_path}` 自动生成；每种模型展示测试 Exact Match 最优的一次运行。",
        ]
    )
    return "\n".join(lines)


def update_readme(
    readme_path: Path = README_PATH,
    index_path: Path = INDEX_PATH,
) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        raise ValueError("README 缺少实验结果生成区域标记")
    before, remainder = text.split(START_MARKER, 1)
    _, after = remainder.split(END_MARKER, 1)
    generated = render_results(index_path)
    readme_path.write_text(
        f"{before}{START_MARKER}\n\n{generated}\n\n{END_MARKER}{after}",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="根据实验索引更新 README")
    parser.add_argument("--readme", type=Path, default=README_PATH)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    args = parser.parse_args()
    update_readme(args.readme, args.index)
    print(f"README updated: {args.readme}")


if __name__ == "__main__":
    main()
