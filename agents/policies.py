"""Cache-eviction policies used by both the HF Space UI and the evaluator.

Each policy has signature ``policy(obs: Observation) -> Action`` so it plugs
directly into ``CDNCacheEnv.step(...)``.
"""

from __future__ import annotations

import random
from typing import Tuple

from env.models import Action, Observation


def lru_baseline(obs: Observation) -> Action:
    """Evict the least recently used file when a miss forces eviction."""
    if obs.cache_hit or not obs.cached_files:
        return Action(evict_file_id=None)
    victim = min(obs.cached_files, key=lambda f: f.last_accessed)
    return Action(evict_file_id=victim.file_id)


def random_baseline(obs: Observation, rng: random.Random | None = None) -> Action:
    """Evict a uniformly random cached file. Sanity-check lower bound."""
    if obs.cache_hit or not obs.cached_files:
        return Action(evict_file_id=None)
    picker = rng or random
    victim = picker.choice(obs.cached_files)
    return Action(evict_file_id=victim.file_id)


def smart_agent(obs: Observation) -> Action:
    """Distilled RL policy with CDN guardrails.

    On every cache miss the agent proposes a victim ranked by:
      1. Not in the short prefetch preview (queue look-ahead).
      2. Not currently viral.
      3. Low request frequency.
      4. Large size (free more room per eviction).

    The env only consumes the ``evict_file_id`` when the incoming file cannot
    fit, so nominating a victim on every miss is strictly >= returning ``None``
    (which would incur a wasted-capacity penalty and skip admission).
    """
    if obs.cache_hit or not obs.cached_files:
        return Action(evict_file_id=None)

    preview = set(obs.queue_preview)

    def score(file_entry) -> Tuple[int, int, float, float]:
        preview_keep = 1 if file_entry.file_id in preview else 0
        viral_keep = 1 if file_entry.is_viral else 0
        return (
            preview_keep,
            viral_keep,
            file_entry.request_frequency,
            -file_entry.size_mb,
        )

    victim = min(obs.cached_files, key=score)
    return Action(evict_file_id=victim.file_id)
