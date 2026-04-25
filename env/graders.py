"""
Deterministic graders for all 3 tasks.
Each grader runs a full episode and returns a score in [0.0, 1.0].
"""

from typing import Callable, Dict, List
from env.cache import CDNCacheEnv, TASK_CONFIGS
from env.models import Action, Observation


GraderPolicy = Callable[[Observation], Action]


def _run_episode(task_id: str, policy: GraderPolicy, seed: int = 42) -> Dict:
    """Run one full episode with a given policy. Returns stats dict."""
    env = CDNCacheEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action = policy(obs)
        result = env.step(action)
        total_reward += result.reward.total
        obs = result.observation
        steps += 1
        if result.done:
            break

    state = env.state()
    return {
        "hit_rate": state["hit_rate"],
        "total_reward": total_reward,
        "bandwidth_saved_mb": state["bandwidth_saved_mb"],
        "steps": steps,
        "hits": state["hits"],
        "misses": state["misses"],
    }


# ─────────────────────────────────────────────
# Built-in Policies (for baseline + grading)
# ─────────────────────────────────────────────

def lru_policy(obs: Observation) -> Action:
    """Evict Least Recently Used file."""
    if not obs.cached_files:
        return Action(evict_file_id=None)
    lru = min(obs.cached_files, key=lambda f: f.last_accessed)
    return Action(evict_file_id=lru.file_id)


def lfu_policy(obs: Observation) -> Action:
    """Evict Least Frequently Used file."""
    if not obs.cached_files:
        return Action(evict_file_id=None)
    lfu = min(obs.cached_files, key=lambda f: f.request_frequency)
    return Action(evict_file_id=lfu.file_id)


def smart_policy(obs: Observation) -> Action:
    """
    Smarter policy:
    - Never evict viral files
    - Evict the lowest-frequency, largest file (wastes least value, frees most space)
    """
    if not obs.cached_files:
        return Action(evict_file_id=None)

    # Filter out viral files from eviction candidates
    candidates = [f for f in obs.cached_files if not f.is_viral]
    if not candidates:
        candidates = obs.cached_files  # fallback: evict anything

    # Score: low frequency = good eviction, large size = good eviction
    def eviction_score(f):
        return -f.request_frequency + f.size_mb * 0.1

    best = max(candidates, key=eviction_score)
    return Action(evict_file_id=best.file_id)


def no_op_policy(obs: Observation) -> Action:
    """Never evict anything (baseline floor)."""
    return Action(evict_file_id=None)


# ─────────────────────────────────────────────
# Grader Functions
# ─────────────────────────────────────────────

def grade_task_easy(policy: GraderPolicy, seed: int = 42) -> float:
    """
    Easy: steady traffic, 100MB cache.
    Score based purely on hit rate.
    >= 0.60 hit rate = 1.0, scales down to 0.0.
    """
    stats = _run_episode("task_easy", policy, seed)
    hit_rate = stats["hit_rate"]

    # Linear scale: 0.0 hit_rate -> 0.0 score, 0.60+ -> 1.0
    score = min(1.0, hit_rate / 0.60)
    return round(score, 4)


def grade_task_medium(policy: GraderPolicy, seed: int = 42) -> float:
    """
    Medium: mixed traffic, viral files.
    Score = weighted combo of hit rate + bandwidth saved.
    """
    stats = _run_episode("task_medium", policy, seed)
    hit_rate = stats["hit_rate"]
    bandwidth = stats["bandwidth_saved_mb"]

    # Normalize bandwidth: assume 500MB = perfect
    bw_score = min(1.0, bandwidth / 500.0)

    # Hit rate: 0.55 = 1.0
    hr_score = min(1.0, hit_rate / 0.55)

    # 70% hit rate, 30% bandwidth
    score = 0.70 * hr_score + 0.30 * bw_score
    return round(score, 4)


def grade_task_hard(policy: GraderPolicy, seed: int = 42) -> float:
    """
    Hard: constrained cache, many viral bursts.
    Score = hit rate + bandwidth + thrash avoidance.
    """
    stats = _run_episode("task_hard", policy, seed)
    hit_rate = stats["hit_rate"]
    bandwidth = stats["bandwidth_saved_mb"]
    total_reward = stats["total_reward"]

    # Hit rate target: 0.45 = 1.0
    hr_score = min(1.0, hit_rate / 0.45)

    # Bandwidth: 400MB = 1.0
    bw_score = min(1.0, bandwidth / 400.0)

    # Reward signal (captures thrash penalties implicitly)
    # Normalize: 200 reward = 1.0
    rw_score = max(0.0, min(1.0, total_reward / 200.0))

    # 50% hit rate, 25% bandwidth, 25% reward quality
    score = 0.50 * hr_score + 0.25 * bw_score + 0.25 * rw_score
    return round(score, 4)


# ─────────────────────────────────────────────
# Master Grader
# ─────────────────────────────────────────────

def run_all_graders(policy: GraderPolicy, seed: int = 42) -> Dict:
    """Run all 3 graders and return scores + summary."""
    easy = grade_task_easy(policy, seed)
    medium = grade_task_medium(policy, seed)
    hard = grade_task_hard(policy, seed)
    overall = round((easy + medium + hard) / 3, 4)

    return {
        "task_easy": easy,
        "task_medium": medium,
        "task_hard": hard,
        "overall": overall,
        "all_in_range": all(0.0 <= s <= 1.0 for s in [easy, medium, hard]),
    }


if __name__ == "__main__":
    print("=== Running Grader Validation ===\n")

    policies = {
        "no_op": no_op_policy,
        "lru": lru_policy,
        "lfu": lfu_policy,
        "smart": smart_policy,
    }

    for name, policy in policies.items():
        results = run_all_graders(policy)
        print(f"Policy: {name}")
        print(f"  Easy:   {results['task_easy']}")
        print(f"  Medium: {results['task_medium']}")
        print(f"  Hard:   {results['task_hard']}")
        print(f"  Overall:{results['overall']}")
        print(f"  Valid:  {results['all_in_range']}\n")
