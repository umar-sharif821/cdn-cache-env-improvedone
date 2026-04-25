"""
Traffic generator for CDN Cache Optimizer.
Simulates realistic web traffic: steady files + viral bursts.
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class FileProfile:
    file_id: str
    size_mb: float
    base_popularity: float   # base request probability
    is_viral: bool = False
    viral_start: int = -1
    viral_duration: int = 0
    viral_peak: float = 0.0


class TrafficGenerator:
    """
    Generates a stream of file requests.
    - Steady files: consistent low-level demand
    - Viral files: spike suddenly, dominate for a window, then die
    """

    def __init__(
        self,
        num_files: int = 50,
        viral_ratio: float = 0.2,
        episode_length: int = 200,
        seed: int = 42,
    ):
        self.num_files = num_files
        self.viral_ratio = viral_ratio
        self.episode_length = episode_length
        self.rng = random.Random(seed)
        self.files: List[FileProfile] = []
        self.request_log: List[str] = []  # precomputed episode
        self._build_file_profiles()
        self._precompute_requests()

    def _build_file_profiles(self):
        num_viral = max(1, int(self.num_files * self.viral_ratio))
        for i in range(self.num_files):
            fid = f"file_{i:03d}"
            size = round(self.rng.uniform(1.0, 20.0), 1)
            is_viral = i < num_viral

            if is_viral:
                viral_start = self.rng.randint(
                    5, max(6, self.episode_length - 30)
                )
                viral_duration = self.rng.randint(10, 30)
                viral_peak = self.rng.uniform(0.4, 0.8)
                base_pop = self.rng.uniform(0.01, 0.05)
                self.files.append(FileProfile(
                    file_id=fid,
                    size_mb=size,
                    base_popularity=base_pop,
                    is_viral=True,
                    viral_start=viral_start,
                    viral_duration=viral_duration,
                    viral_peak=viral_peak,
                ))
            else:
                base_pop = self.rng.uniform(0.02, 0.15)
                self.files.append(FileProfile(
                    file_id=fid,
                    size_mb=size,
                    base_popularity=base_pop,
                ))

    def _get_popularity_at_step(self, fp: FileProfile, step: int) -> float:
        if not fp.is_viral:
            # Steady with slight daily cycle
            cycle = 0.3 * math.sin(2 * math.pi * step / 50)
            return max(0.001, fp.base_popularity + cycle * fp.base_popularity)

        # Viral: bell curve spike
        if step < fp.viral_start or step > fp.viral_start + fp.viral_duration:
            return fp.base_popularity
        center = fp.viral_start + fp.viral_duration / 2
        spread = fp.viral_duration / 4
        spike = fp.viral_peak * math.exp(-((step - center) ** 2) / (2 * spread ** 2))
        return fp.base_popularity + spike

    def _precompute_requests(self):
        self.request_log = []
        for step in range(self.episode_length):
            weights = [
                self._get_popularity_at_step(fp, step) for fp in self.files
            ]
            total = sum(weights)
            norm = [w / total for w in weights]
            chosen = self.rng.choices(self.files, weights=norm, k=1)[0]
            self.request_log.append(chosen.file_id)

    def get_request(self, step: int) -> Tuple[str, float, bool]:
        """Returns (file_id, size_mb, is_viral) for a given step."""
        if step >= len(self.request_log):
            return self.request_log[-1], 1.0, False
        fid = self.request_log[step]
        fp = next(f for f in self.files if f.file_id == fid)
        return fid, fp.size_mb, fp.is_viral

    def get_preview(self, step: int, n: int = 3) -> List[str]:
        """Peek at next n file_ids (simulates prefetch hints)."""
        return self.request_log[step + 1: step + 1 + n]

    def get_file_profile(self, file_id: str) -> FileProfile:
        return next((f for f in self.files if f.file_id == file_id), None)

    def time_of_day(self, step: int) -> float:
        """Normalized 0.0–1.0 cycle."""
        return (step % 50) / 50.0
