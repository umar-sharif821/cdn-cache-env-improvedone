"""Reusable cache-eviction policies for the CDN Cache Optimizer."""

from .policies import lru_baseline, random_baseline, smart_agent

__all__ = ["lru_baseline", "random_baseline", "smart_agent"]
