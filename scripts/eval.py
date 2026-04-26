"""Reproducible evaluator for judges.

Usage:
    python scripts/eval.py               # full 3-task x 3-seed sweep
    python scripts/eval.py --quick       # 1 seed per task (fast CI smoke)
    python scripts/eval.py --out out.json

Outputs a markdown table to stdout and writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.policies import lru_baseline, random_baseline, smart_agent  # noqa: E402
from env.cache import CDNCacheEnv, TASK_CONFIGS  # noqa: E402
from env.models import Action, Observation  # noqa: E402


POLICIES: Dict[str, Callable[[Observation], Action]] = {
    "random": lambda obs: random_baseline(obs, rng=random.Random(0)),
    "lru_baseline": lru_baseline,
    "smart_agent": smart_agent,
}


def run_episode(task_id: str, seed: int, policy: Callable[[Observation], Action]) -> Dict:
    env = CDNCacheEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    rewards: List[float] = []
    info: Dict = {}
    done = False
    while not done:
        result = env.step(policy(obs))
        obs = result.observation
        info = result.info
        rewards.append(float(result.reward.total))
        done = result.done
    return {
        "task_id": task_id,
        "seed": seed,
        "total_reward": float(sum(rewards)),
        "final_hit_rate": float(info.get("hit_rate", 0.0)),
        "bandwidth_saved_mb": float(info.get("bandwidth_saved_mb", 0.0)),
    }


def summarize(runs: List[Dict]) -> Dict:
    reward = [r["total_reward"] for r in runs]
    hit = [r["final_hit_rate"] for r in runs]
    bw = [r["bandwidth_saved_mb"] for r in runs]
    return {
        "reward_mean": statistics.mean(reward),
        "reward_std": statistics.pstdev(reward) if len(reward) > 1 else 0.0,
        "hit_rate_mean": statistics.mean(hit),
        "bandwidth_mean": statistics.mean(bw),
        "n": len(runs),
    }


def format_markdown(results: Dict) -> str:
    header = "| Task | Policy | Hit Rate | Reward (mean +/- std) | Bandwidth MB |\n"
    header += "|---|---|---|---|---|\n"
    rows = []
    for task_id in results["tasks"]:
        for policy_name in POLICIES.keys():
            s = results["summary"][task_id][policy_name]
            rows.append(
                f"| {task_id} | {policy_name} | {s['hit_rate_mean']:.1%} | "
                f"{s['reward_mean']:.2f} +/- {s['reward_std']:.2f} | "
                f"{s['bandwidth_mean']:.1f} |"
            )
    return header + "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="CDN Cache Optimizer evaluator")
    parser.add_argument("--quick", action="store_true", help="1 seed per task (fast)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="seeds to run")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_CONFIGS.keys()))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval_results.json")
    args = parser.parse_args()

    seeds = [0] if args.quick else args.seeds
    t0 = time.time()

    runs: List[Dict] = []
    summary: Dict[str, Dict[str, Dict]] = {}
    for task_id in args.tasks:
        summary[task_id] = {}
        for policy_name, policy_fn in POLICIES.items():
            per_task_runs = []
            for seed in seeds:
                rec = run_episode(task_id, seed, policy_fn)
                rec["policy"] = policy_name
                runs.append(rec)
                per_task_runs.append(rec)
            summary[task_id][policy_name] = summarize(per_task_runs)

    elapsed = time.time() - t0
    results = {
        "tasks": args.tasks,
        "seeds": seeds,
        "runs": runs,
        "summary": summary,
        "elapsed_sec": round(elapsed, 2),
    }

    args.out.write_text(json.dumps(results, indent=2))
    print(format_markdown(results))
    print(f"\nWrote {args.out} ({elapsed:.1f}s, {len(runs)} episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
