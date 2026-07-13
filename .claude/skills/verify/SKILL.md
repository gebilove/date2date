---
name: verify
summary: Verify date2date models through the experiment CLI.
---

# Verify date2date experiments

Run the real CLI from the repository root. Use the existing fixed test set to avoid creating another tracked fixture:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m date2date.experiment \
  --model <model> --hidden-size 8 --attention-heads 2 --lr 0.01 \
  --train-size 16 --test-size 100 --epochs 1 --batch-size 8 \
  --seed 42 --test-seed 20260710 --no-update-readme \
  --notes "temporary smoke: <model>"
```

Capture the printed run ID. Check its `config.json`, `metrics.json`, and `checkpoint.pt`; require finite metrics and the expected model configuration. Probe invalid CLI values such as `--attention-heads 0` and confirm argparse exits before creating a run.

Smoke runs append `experiments/experiments.csv` and create `experiments/runs/<run-id>`. After collecting evidence, remove only the captured run directories and their exact CSV rows so smoke metrics do not enter the README leaderboard.
