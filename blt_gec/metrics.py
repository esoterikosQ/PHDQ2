"""Metric helpers shared by BLT training and standalone evaluation."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from baseline.metric.gleumodule import run_gleu


def compute_gleu(reference: str | Path, source: str | Path, hypothesis: str | Path) -> float:
    """Return GLEU on the same 0-100 scale used in training logs."""

    return float(
        run_gleu(
            reference=str(reference),
            source=str(source),
            hypothesis=str(hypothesis),
        )
    ) * 100


def compute_m2(
    hypothesis_path: str | Path,
    source_gold_path: str | Path | None,
) -> tuple[float | None, float | None, float | None]:
    if not source_gold_path:
        return None, None, None

    source_gold = Path(source_gold_path)
    if not source_gold.exists():
        print(f"Warning: M2 source-gold file does not exist: {source_gold}")
        return None, None, None

    scorer = Path("baseline/metric/m2scorer/scripts/m2scorer.py")
    cmd = [sys.executable, str(scorer), str(hypothesis_path), str(source_gold)]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Warning: M2 scorer failed: {exc.stderr or exc.stdout}")
        return None, None, None

    metrics = {}
    for line in completed.stdout.splitlines():
        match = re.match(r"(Precision|Recall|F_0\.5)\s*:\s*([0-9.]+)", line.strip())
        if match:
            metrics[match.group(1)] = float(match.group(2))
    if not {"Precision", "Recall", "F_0.5"} <= set(metrics):
        print(f"Warning: could not parse M2 scorer output: {completed.stdout}")
        return None, None, None
    return metrics["Precision"], metrics["Recall"], metrics["F_0.5"]
