"""
Core CDN Cache simulation.
Implements full OpenEnv interface: reset(), step(), state()
"""

from collections import defaultdict
from typing import Dict, Optional, List, Tuple
from env.models import (
    Observation, Action, Reward, StepResult, FileEntry, TaskConfig
)
from env.traffic import TrafficGenerator


TASK_CONFIGS = {
    "task_easy": TaskConfig(
        task_id="task_easy",
        name="Steady Traffic Cache",
        difficulty="easy",
        cache_capacity_mb=100.0,
        num_files=30,
        viral_ratio=0.0,         # no viral files
        episode_length=100,
        description=(
            "Cache has 100MB capacity. Only steady traffic files. "
            "Agent must learn LRU-style eviction. Target hit rate >= 0.60."
        ),
    ),
    "task_medium": TaskConfig(
        task_id="task_medium",
        name="Mixed Traffic Cache",
        difficulty="medium",
        cache_capacity_mb=80.0,
        num_files=50,
        viral_ratio=0.2,
        episode_length=150,
        description=(
            "80MB cache, mix of steady and viral files. "
            "Agent must prioritize popular content and handle viral spikes. "
            "Target hit rate >= 0.55 with efficient eviction."
        ),
    ),
    "task_hard": TaskConfig(
        task_id="task_hard",
        name="Constrained Cache with Viral Bursts",
        difficulty="hard",
        cache_capacity_mb=50.0,
        num_files=80,
        viral_ratio=0.35,
        episode_length=200,
        description=(
            "Tight 50MB cache, many viral bursts, large file sizes. "
            "Agent must predict spikes, avoid cache thrashing, "
            "and maximize bandwidth saved. Target hit rate >= 0.45."
        ),
    ),
}


class CDNCacheEnv:
    """
    CDN Cache Optimizer Environment.
    At each step, a file is requested. If not cached, agent must decide
    which file (if any) to evict to make room for the new one.
    """

    def __init__(self, task_id: str = "task_easy", seed: int = 42):
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id: {task_id}. Choose from {list(TASK_CONFIGS.keys())}")
        self.config = TASK_CONFIGS[task_id]
        self.seed = seed
        self._cache: Dict[str, FileEntry] = {}       # file_id -> FileEntry
        self._cache_used_mb: float = 0.0
        self._step: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._recent_hits: List[bool] = []
        self._last_evicted: Optional[str] = None
        self._eviction_counts: Dict[str, int] = defaultdict(int)
        self._total_bandwidth_saved: float = 0.0
        self._done: bool = False
        self.traffic = TrafficGenerator(
            num_files=self.config.num_files,
            viral_ratio=self.config.viral_ratio,
            episode_length=self.config.episode_length,
            seed=seed,
        )

    # ─────────────────────────────────────────────
    # OpenEnv Interface
    # ─────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment to initial state."""
        self._cache = {}
        self._cache_used_mb = 0.0
        self._step = 0
        self._hits = 0
        self._misses = 0
        self._recent_hits = []
        self._last_evicted = None
        self._eviction_counts = defaultdict(int)
        self._total_bandwidth_saved = 0.0
        self._done = False
        self.traffic = TrafficGenerator(
            num_files=self.config.num_files,
            viral_ratio=self.config.viral_ratio,
            episode_length=self.config.episode_length,
            seed=self.seed,
        )
        return self._make_observation(cache_hit=False)

    def step(self, action: Action) -> StepResult:
        """Process one step: handle eviction, then serve the request."""
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")

        file_id, size_mb, is_viral = self.traffic.get_request(self._step)
        cache_hit = file_id in self._cache
        reward = self._process_step(action, file_id, size_mb, is_viral, cache_hit)

        self._step += 1
        self._done = self._step >= self.config.episode_length

        obs = self._make_observation(cache_hit=cache_hit)
        info = {
            "total_hits": self._hits,
            "total_misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
            "cache_fill_ratio": self._cache_used_mb / self.config.cache_capacity_mb,
            "bandwidth_saved_mb": self._total_bandwidth_saved,
        }
        return StepResult(observation=obs, reward=reward, done=self._done, info=info)

    def state(self) -> dict:
        """Return current full environment state."""
        return {
            "step": self._step,
            "done": self._done,
            "cache": {k: v.dict() for k, v in self._cache.items()},
            "cache_used_mb": self._cache_used_mb,
            "cache_capacity_mb": self.config.cache_capacity_mb,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
            "bandwidth_saved_mb": self._total_bandwidth_saved,
            "task": self.config.dict(),
        }

    # ─────────────────────────────────────────────
    # Internal Logic
    # ─────────────────────────────────────────────

    def _process_step(
        self,
        action: Action,
        file_id: str,
        size_mb: float,
        is_viral: bool,
        cache_hit: bool,
    ) -> Reward:
        hit_bonus = 0.0
        eviction_penalty = 0.0
        thrash_penalty = 0.0
        bandwidth_saved = 0.0
        wasted_penalty = 0.0

        if cache_hit:
            self._hits += 1
            self._recent_hits.append(True)
            hit_bonus = 1.0 + (0.5 if is_viral else 0.0)   # viral hits worth more
            bandwidth_saved = size_mb * 0.01               # normalized
            self._total_bandwidth_saved += size_mb
            # Update frequency
            entry = self._cache[file_id]
            entry.request_frequency = min(entry.request_frequency + 1, 50)
            entry.last_accessed = self._step
        else:
            self._misses += 1
            self._recent_hits.append(False)

            # Try to insert new file
            if self._cache_used_mb + size_mb <= self.config.cache_capacity_mb:
                # Fits without eviction
                self._insert_file(file_id, size_mb, is_viral)
            else:
                # Need to evict
                if action.evict_file_id and action.evict_file_id in self._cache:
                    evicted = self._cache[action.evict_file_id]

                    # Penalize evicting high-frequency files
                    if evicted.request_frequency > 10:
                        eviction_penalty -= 0.3
                    if evicted.is_viral:
                        eviction_penalty -= 0.2

                    # Thrash penalty: evicted and re-requested soon
                    if action.evict_file_id == self._last_evicted:
                        thrash_penalty = -0.5

                    self._eviction_counts[action.evict_file_id] += 1
                    self._remove_file(action.evict_file_id)
                    self._last_evicted = action.evict_file_id

                    if self._cache_used_mb + size_mb <= self.config.cache_capacity_mb:
                        self._insert_file(file_id, size_mb, is_viral)
                else:
                    # No valid eviction action — wasted capacity penalty
                    wasted_penalty = -0.2

        # Wasted capacity: cache too empty when we could be caching
        fill_ratio = self._cache_used_mb / self.config.cache_capacity_mb
        if fill_ratio < 0.3 and self._step > 10:
            wasted_penalty -= 0.1

        # Keep recent_hits window at 20
        if len(self._recent_hits) > 20:
            self._recent_hits.pop(0)

        total = hit_bonus + eviction_penalty + thrash_penalty + bandwidth_saved + wasted_penalty
        return Reward(
            total=round(total, 4),
            cache_hit_bonus=hit_bonus,
            eviction_penalty=eviction_penalty,
            thrash_penalty=thrash_penalty,
            bandwidth_saved=bandwidth_saved,
            wasted_capacity_penalty=wasted_penalty,
        )

    def _insert_file(self, file_id: str, size_mb: float, is_viral: bool):
        self._cache[file_id] = FileEntry(
            file_id=file_id,
            size_mb=size_mb,
            request_frequency=1.0,
            is_viral=is_viral,
            last_accessed=self._step,
        )
        self._cache_used_mb += size_mb

    def _remove_file(self, file_id: str):
        if file_id in self._cache:
            self._cache_used_mb -= self._cache[file_id].size_mb
            self._cache_used_mb = max(0.0, self._cache_used_mb)
            del self._cache[file_id]

    def _make_observation(self, cache_hit: bool) -> Observation:
        file_id, size_mb, is_viral = self.traffic.get_request(self._step)
        preview = self.traffic.get_preview(self._step)
        recent_hit_rate = (
            sum(self._recent_hits) / len(self._recent_hits)
            if self._recent_hits else 0.0
        )
        fill = self._cache_used_mb / self.config.cache_capacity_mb
        return Observation(
            step=self._step,
            cache_used_mb=round(self._cache_used_mb, 2),
            cache_capacity_mb=self.config.cache_capacity_mb,
            cache_fill_ratio=round(fill, 4),
            cached_files=list(self._cache.values()),
            incoming_file_id=file_id,
            incoming_file_size_mb=size_mb,
            incoming_file_is_viral=is_viral,
            cache_hit=cache_hit,
            recent_hit_rate=round(recent_hit_rate, 4),
            time_of_day=round(self.traffic.time_of_day(self._step), 4),
            queue_preview=preview,
        )
class DriftCDNEnv(CDNCacheEnv):
    def __init__(self, task_id="task_hard", seed=42):
        super().__init__(task_id=task_id, seed=seed)
        self._original_capacity = self.config.cache_capacity_mb
        self._hit_multiplier = 1.0
        self._thrash_multiplier = 1.0
    def reset(self):
        obs = super().reset()
        self.config.cache_capacity_mb = self._original_capacity
        self._hit_multiplier = 1.0
        self._thrash_multiplier = 1.0
        return obs
    def step(self, action):
        self._apply_drift()
        result = super().step(action)
        r = result.reward
        new_total = round(r.cache_hit_bonus*self._hit_multiplier + r.eviction_penalty + r.thrash_penalty*self._thrash_multiplier + r.bandwidth_saved + r.wasted_capacity_penalty, 4)
        from env.models import Reward, StepResult
        return StepResult(observation=result.observation, reward=Reward(total=new_total, cache_hit_bonus=r.cache_hit_bonus*self._hit_multiplier, eviction_penalty=r.eviction_penalty, thrash_penalty=r.thrash_penalty*self._thrash_multiplier, bandwidth_saved=r.bandwidth_saved, wasted_capacity_penalty=r.wasted_capacity_penalty), done=result.done, info=result.info)
    def _apply_drift(self):
        if self._step == 50:
            self.config.cache_capacity_mb *= 0.6
            self._cache_used_mb = min(self._cache_used_mb, self.config.cache_capacity_mb)
        elif self._step == 100:
            self.traffic.viral_ratio = min(1.0, self.traffic.viral_ratio + 0.25)
        elif self._step == 150:
            self._hit_multiplier = 0.6
            self._thrash_multiplier = 2.5