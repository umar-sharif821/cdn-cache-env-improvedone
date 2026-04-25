"""
Typed Pydantic models for the CDN Cache Optimizer environment.
Implements OpenEnv spec: Observation, Action, Reward.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class FileEntry(BaseModel):
    """Represents a file currently in the cache."""
    file_id: str
    size_mb: float
    request_frequency: float   # requests per last N steps
    is_viral: bool
    last_accessed: int         # step number


class Observation(BaseModel):
    """What the agent sees at each step."""
    step: int
    cache_used_mb: float
    cache_capacity_mb: float
    cache_fill_ratio: float
    cached_files: List[FileEntry]
    incoming_file_id: str
    incoming_file_size_mb: float
    incoming_file_is_viral: bool
    cache_hit: bool                     # was incoming_file already cached?
    recent_hit_rate: float              # rolling hit rate last 20 steps
    time_of_day: float                  # 0.0 to 1.0 (normalized)
    queue_preview: List[str]            # next 3 file_ids coming


class Action(BaseModel):
    """What the agent decides to do."""
    evict_file_id: Optional[str] = None   # None = do nothing / already cached


class Reward(BaseModel):
    """Reward breakdown for transparency."""
    total: float
    cache_hit_bonus: float
    eviction_penalty: float
    thrash_penalty: float
    bandwidth_saved: float
    wasted_capacity_penalty: float


class StepResult(BaseModel):
    """Full result returned by step()."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict


class TaskConfig(BaseModel):
    """Configuration for a specific task."""
    task_id: str
    name: str
    difficulty: str
    cache_capacity_mb: float
    num_files: int
    viral_ratio: float
    episode_length: int
    description: str
