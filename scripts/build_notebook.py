"""Convert colab_submission_script.py into a clean Colab .ipynb notebook.

Splits the script on the `# === ... STEP N ...` banner blocks and emits one
code cell per step, with a markdown intro cell at the top.

Usage:
    python scripts/build_notebook.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "colab_submission_script.py"
OUTPUT = REPO_ROOT / "notebooks" / "cdn_cache_optimizer_training.ipynb"

INTRO_MD = """\
# CDN Cache Optimizer — Training Notebook

OpenEnv-compliant reinforcement-learning agent for **edge CDN cache admission and eviction**.
Run **Runtime → Run all** in Colab to reproduce training, evaluation, schema-drift verification, and result charts in a single pass.

**Project links**
- Hugging Face Space: https://huggingface.co/spaces/umar-sharif821/cdn-cache-env-improvedone
- GitHub repo: https://github.com/umar-sharif821/cdn-cache-env-improvedone

**What this notebook does**
1. Bootstraps Colab (installs `gymnasium`, `torch`, `matplotlib`, `numpy`; mounts Drive if available).
2. Defines a `SchemaDriftGuard` that normalizes heterogeneous CDN log formats.
3. Builds an OpenEnv-compliant `CDNCacheEnv` (gymnasium 5-tuple, multi-component reward).
4. Trains a REINFORCE policy network.
5. Evaluates LRU baseline vs. the fine-tuned agent.
6. Saves `policy.pt`, `training_results.png`, `drift_report.json`, `metrics.json`.

**Reward function**
`R = w1 * Perf - w2 * Cost`, where `Perf` is edge-vs-origin latency savings and `Cost` is eviction churn + admitted bytes / capacity.
"""

STEP_TITLES = {
    0: "Step 0 — Colab bootstrap (deps + Drive)",
    1: "Step 1 — Imports & deterministic seeding",
    2: "Step 2 — Schema Drift Guard",
    3: "Step 3 — OpenEnv-compliant CDN cache environment",
    4: "Step 4 — Policy network + REINFORCE training loop",
    5: "Step 5 — Evaluation: LRU baseline vs fine-tuned agent",
    6: "Step 6 — Comparison charts",
    7: "Step 7 — Persist artifacts to Drive",
    8: "Step 8 — Submission summary",
}


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


def make_md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def split_into_steps(text: str) -> list[tuple[int, str]]:
    """Return (step_index, body_without_banner) tuples in order."""
    banner = re.compile(r"# ={5,}\n# STEP (\d+)[^\n]*\n# ={5,}\n")
    matches = list(banner.finditer(text))
    if not matches:
        raise RuntimeError("No STEP banners found in source script.")

    steps: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        step_idx = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip("\n")
        steps.append((step_idx, body))
    return steps


def build_notebook() -> dict:
    raw = SOURCE.read_text(encoding="utf-8")
    docstring_match = re.match(r'"""(.*?)"""', raw, flags=re.DOTALL)
    if docstring_match:
        body = raw[docstring_match.end():].lstrip("\n")
    else:
        body = raw

    steps = split_into_steps(body)

    cells: list[dict] = [make_md_cell(INTRO_MD)]
    for step_idx, code in steps:
        title = STEP_TITLES.get(step_idx, f"Step {step_idx}")
        cells.append(make_md_cell(f"## {title}"))
        cells.append(make_code_cell(code))

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
            "colab": {"provenance": []},
        },
        "cells": cells,
    }


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    OUTPUT.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
