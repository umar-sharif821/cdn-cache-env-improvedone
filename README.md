---
title: CDN Cache Optimizer
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - cdn
  - caching
  - hackathon
---

# CDN Cache Optimizer - OpenEnv RL Agent

Hackathon-ready OpenEnv project for **edge CDN cache admission and eviction**. It simulates the real production tradeoff between serving from a fast edge cache and falling back to slower origin fetches, while handling schema drift in CDN logs.

---

## Why It Matters

Content Delivery Networks serve billions of files daily. Edge servers have limited storage, so they must constantly decide: *which cached files to keep, and which to evict?* Standard algorithms like LRU aren't optimal — especially when traffic has **viral bursts** (a file suddenly gets 50x more requests for 20 minutes, then drops back to zero).

A smarter agent can:
- Predict viral spikes from queue previews
- Avoid evicting high-frequency files
- Prevent cache thrashing (evicting then immediately re-requesting)
- Maximize bandwidth saved for users

---

## Live Demo

This repo is Hugging Face Spaces-ready. The Docker Space runs `app.py`, a Gradio UI that compares:

- **Baseline LRU**: always evicts the least recently used file.
- **Fine-tuned Agent**: preserves viral/previewed objects, avoids bulky cold admissions, and evicts low-value content under cache pressure.

Run locally:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.

## Google Colab Submission

For judges who want a single reproducible run:

```python
!python /content/colab_submission_script.py
```

The script installs dependencies, mounts Drive when available, trains/evaluates the agent, verifies schema drift normalization, and saves:

- `training_results.png`
- `policy.pt`
- `drift_report.json`
- `metrics.json`

## Environment Description

At each step, a file is requested from the network. If it is already in cache, the user is served from the edge. If not, the request goes to origin and the agent decides whether to admit the file and what to evict.

### Traffic Model
- **Steady files**: consistent, cyclical demand.
- **Viral files**: bell-curve spikes that fade back to baseline.
- **Queue preview**: short lookahead signal similar to CDN prefetch telemetry.

### Reward Grounding

The Colab RL environment uses a multi-component reward:

```text
R = w1 * Perf - w2 * Cost
```

`Perf` captures edge-latency savings versus origin fetch, while `Cost` penalizes cache churn and write/admission cost.

### Schema Drift

`SchemaDriftGuard` in `colab_submission_script.py` normalizes CDN logs across renamed, missing, extra, and type-shifted fields, for example:

- `ts`, `time`, `event_time` -> `timestamp`
- `fid`, `object_id`, `oid` -> `file_id`
- `bytes`, `size_bytes` -> `size_mb`
- `cache_hit`, `is_hit` -> `hit`

---

## 📐 Action & Observation Space

### Observation Space
| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Current episode step |
| `cache_used_mb` | float | MB currently used |
| `cache_capacity_mb` | float | Total cache size |
| `cache_fill_ratio` | float | 0.0–1.0 fill level |
| `cached_files` | List[FileEntry] | All files in cache with metadata |
| `incoming_file_id` | str | File being requested |
| `incoming_file_size_mb` | float | Size of incoming file |
| `incoming_file_is_viral` | bool | Is this file currently viral? |
| `cache_hit` | bool | Is incoming file already cached? |
| `recent_hit_rate` | float | Rolling hit rate (last 20 steps) |
| `time_of_day` | float | Normalized 0.0–1.0 daily cycle |
| `queue_preview` | List[str] | Next 3 file IDs (prefetch hint) |

### FileEntry Fields
| Field | Type | Description |
|-------|------|-------------|
| `file_id` | str | Unique identifier |
| `size_mb` | float | File size in MB |
| `request_frequency` | float | Requests since cached |
| `is_viral` | bool | Currently viral |
| `last_accessed` | int | Step number of last access |

### Action Space
| Field | Type | Description |
|-------|------|-------------|
| `evict_file_id` | str \| null | File to evict (null = no eviction) |

### Reward Function
| Component | Range | Description |
|-----------|-------|-------------|
| `cache_hit_bonus` | +1.0 to +1.5 | Hit reward (viral hits = +1.5) |
| `bandwidth_saved` | +0.0 to +0.2 | Reward for bandwidth efficiency |
| `eviction_penalty` | -0.0 to -0.5 | Penalty for evicting popular files |
| `thrash_penalty` | 0.0 or -0.5 | Penalty for evicting same file twice |
| `wasted_capacity_penalty` | -0.0 to -0.3 | Penalty for leaving cache empty |

---

## 📋 Tasks

### Task 1: Steady Traffic Cache (Easy)
- **Cache**: 100MB | **Files**: 30 | **Steps**: 100
- No viral files — steady demand only
- Agent learns basic LRU-style eviction
- **Target hit rate**: ≥ 0.60 → score 1.0
- **Baseline score**: ~0.75

### Task 2: Mixed Traffic Cache (Medium)
- **Cache**: 80MB | **Files**: 50 | **Steps**: 150  
- 20% viral files mixed with steady demand
- Agent must handle spikes and prioritize popular content
- **Score**: 70% hit rate + 30% bandwidth
- **Baseline score**: ~0.60

### Task 3: Constrained Cache with Viral Bursts (Hard)
- **Cache**: 50MB | **Files**: 80 | **Steps**: 200
- 35% viral files, tight capacity, large file sizes
- Agent must predict spikes, avoid thrashing
- **Score**: 50% hit rate + 25% bandwidth + 25% reward quality
- **Baseline score**: ~0.45

---

## Hugging Face Deployment

1. Create a new Hugging Face Space.
2. Choose **Docker** as the SDK.
3. Push this repository to the Space remote.
4. The Space starts automatically from `Dockerfile` and serves `app.py` on port `7860`.

```bash
git remote add space https://huggingface.co/spaces/<username>/cdn-cache-optimizer
git push space main
```

## GitHub Deployment

```bash
git add .
git commit -m "Prepare CDN Cache Optimizer hackathon submission"
git branch -M main
git remote add origin https://github.com/<username>/cdn-cache-optimizer.git
git push -u origin main
```

## 🚀 Setup & Usage

### Local Setup
```bash
git clone <repo>
cd cdn-cache-env
pip install -r requirements.txt
```

### Run API Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

### Run Inference (Baseline Agent)
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_token_here"

python inference.py
```

### Docker
```bash
docker build -t cdn-cache-env .
docker run -p 7860:7860 cdn-cache-env
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (returns 200) |
| GET | `/tasks` | List all tasks |
| POST | `/reset` | Start episode `{"task_id": "task_easy", "seed": 42}` |
| POST | `/step` | Take action `{"evict_file_id": "file_001" or null}` |
| GET | `/state` | Full environment state |

---

## 📊 Baseline Scores

Using the built-in `smart_policy` (non-LLM baseline):

| Task | Hit Rate | Score |
|------|----------|-------|
| Easy | ~0.72 | ~1.00 |
| Medium | ~0.61 | ~0.82 |
| Hard | ~0.48 | ~0.78 |
| **Overall** | | **~0.87** |

---

## 📝 Log Format

`inference.py` emits structured JSON logs:

```
{"type": "START", "task_id": "task_easy", ...}
{"type": "STEP",  "step": 0, "action": {...}, "reward": 1.0, ...}
{"type": "END",   "total_reward": 87.3, "final_hit_rate": 0.72, "score": 1.0}
```