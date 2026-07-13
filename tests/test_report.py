import csv
import tempfile
import unittest
from pathlib import Path

from date2date.report import _best_runs, render_results


class ExperimentReportTest(unittest.TestCase):
    def _row(
        self,
        model: str,
        batch_size: int,
        exact_match: float,
        date: str,
    ) -> dict[str, str]:
        return {
            "model_type": model,
            "batch_size": str(batch_size),
            "test_size": "100",
            "test_exact_match": str(exact_match),
            "test_char_accuracy": str(exact_match),
            "parameter_count": "1000",
            "duration_seconds": "1.5",
            "date": date,
        }

    def test_best_runs_keeps_each_model_and_batch_combination(self):
        rows = [
            self._row("vanilla_gru", 32, 0.8, "2026-07-13"),
            self._row("vanilla_gru", 1, 0.7, "2026-07-10"),
            self._row("vanilla_gru", 32, 0.9, "2026-07-14"),
            self._row("bahdanau", 32, 1.0, "2026-07-13"),
        ]

        best = _best_runs(rows)

        self.assertEqual(
            [(row["model_type"], row["batch_size"]) for row in best],
            [("bahdanau", "32"), ("vanilla_gru", "1"), ("vanilla_gru", "32")],
        )
        self.assertEqual(best[-1]["test_exact_match"], "0.9")

    def test_render_results_includes_batch_column(self):
        rows = [self._row("vanilla_gru", 1, 0.7, "2026-07-10")]
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "experiments.csv"
            with index_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)

            rendered = render_results(index_path)

        self.assertIn("| 模型 | Batch |", rendered)
        self.assertIn("| vanilla_gru | 1 | 100 |", rendered)


if __name__ == "__main__":
    unittest.main()
