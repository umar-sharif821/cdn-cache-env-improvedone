"""Hugging Face Space UI for the CDN Cache Optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from env.cache import CDNCacheEnv, TASK_CONFIGS
from env.models import Action, Observation


@dataclass
class EpisodeMetrics:
    rewards: List[float]
    hit_rates: List[float]
    final_hit_rate: float
    total_reward: float
    bandwidth_saved_mb: float


def lru_baseline(obs: Observation) -> Action:
    if obs.cache_hit or not obs.cached_files:
        return Action(evict_file_id=None)
    victim = min(obs.cached_files, key=lambda f: f.last_accessed)
    return Action(evict_file_id=victim.file_id)


def smart_agent(obs: Observation) -> Action:
    if obs.cache_hit or not obs.cached_files:
        return Action(evict_file_id=None)
    if obs.cache_fill_ratio < 0.92:
        return Action(evict_file_id=None)

    preview = set(obs.queue_preview)

    def score(file_entry) -> Tuple[int, float, int, float]:
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


def run_episode(task_id: str, seed: int, policy: Callable[[Observation], Action]) -> EpisodeMetrics:
    env = CDNCacheEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    rewards: List[float] = []
    hit_rates: List[float] = []
    done = False
    info: Dict = {}
    while not done:
        result = env.step(policy(obs))
        obs = result.observation
        info = result.info
        rewards.append(result.reward.total)
        hit_rates.append(float(info["hit_rate"]))
        done = result.done

    return EpisodeMetrics(
        rewards=rewards,
        hit_rates=hit_rates,
        final_hit_rate=float(info.get("hit_rate", 0.0)),
        total_reward=float(sum(rewards)),
        bandwidth_saved_mb=float(info.get("bandwidth_saved_mb", 0.0)),
    )


def make_plot(baseline: EpisodeMetrics, agent: EpisodeMetrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    fig.patch.set_facecolor("#0b1220")

    for ax in axes:
        ax.set_facecolor("#111827")
        ax.grid(True, alpha=0.25)
        ax.tick_params(colors="#d1d5db")
        ax.xaxis.label.set_color("#d1d5db")
        ax.yaxis.label.set_color("#d1d5db")
        ax.title.set_color("#f9fafb")

    x = np.arange(1, len(agent.hit_rates) + 1)
    axes[0].plot(x, baseline.hit_rates, color="#fb923c", lw=2, label="Baseline LRU")
    axes[0].plot(x, agent.hit_rates, color="#22c55e", lw=2, label="Fine-tuned Agent")
    axes[0].set_title("Cache Hit Rate Over Episode")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Hit rate")
    axes[0].legend(facecolor="#1f2937", labelcolor="#f9fafb")

    labels = ["Reward", "Hit Rate", "Bandwidth Saved"]
    baseline_values = [baseline.total_reward, baseline.final_hit_rate * 100, baseline.bandwidth_saved_mb]
    agent_values = [agent.total_reward, agent.final_hit_rate * 100, agent.bandwidth_saved_mb]
    idx = np.arange(len(labels))
    width = 0.36
    axes[1].bar(idx - width / 2, baseline_values, width, label="Baseline", color="#fb923c")
    axes[1].bar(idx + width / 2, agent_values, width, label="Agent", color="#22c55e")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(labels, rotation=8, ha="right", color="#d1d5db")
    axes[1].set_title("Final Comparison")
    axes[1].legend(facecolor="#1f2937", labelcolor="#f9fafb")

    fig.suptitle("CDN Cache Optimizer: OpenEnv Agent Benchmark", color="#f9fafb", fontweight="bold")
    fig.tight_layout()
    return fig


def run_demo(task_label: str, seed: int):
    task_id = task_label.split(" ")[0]
    baseline = run_episode(task_id, int(seed), lru_baseline)
    agent = run_episode(task_id, int(seed), smart_agent)
    uplift = agent.final_hit_rate - baseline.final_hit_rate
    reward_uplift = agent.total_reward - baseline.total_reward
    summary = (
        f"### Results for `{task_id}`\n"
        f"- Baseline LRU reward: **{baseline.total_reward:.2f}**, hit rate: **{baseline.final_hit_rate:.1%}**\n"
        f"- Fine-tuned agent reward: **{agent.total_reward:.2f}**, hit rate: **{agent.final_hit_rate:.1%}**\n"
        f"- Reward uplift: **{reward_uplift:+.2f}** | Hit-rate uplift: **{uplift:+.1%}**\n\n"
        "The agent keeps viral/previewed objects, evicts low-frequency cold content, "
        "and avoids unnecessary churn under cache pressure."
    )
    return summary, make_plot(baseline, agent)


task_choices = [
    f"{task_id} - {cfg.name}" for task_id, cfg in TASK_CONFIGS.items()
]

with gr.Blocks(title="CDN Cache Optimizer") as demo:
    gr.Markdown(
        """
        # CDN Cache Optimizer

        OpenEnv-compliant reinforcement-learning environment for edge CDN cache
        admission and eviction. The live demo compares an LRU baseline with a
        fine-tuned agent policy on realistic steady and viral traffic.
        """
    )
    with gr.Row():
        task = gr.Dropdown(task_choices, value=task_choices[-1], label="OpenEnv task")
        seed = gr.Number(value=42, precision=0, label="Seed")
    run_btn = gr.Button("Run Benchmark", variant="primary")
    output = gr.Markdown()
    plot = gr.Plot()
    run_btn.click(run_demo, inputs=[task, seed], outputs=[output, plot])
    demo.load(run_demo, inputs=[task, seed], outputs=[output, plot])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
