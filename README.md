# ECO — Experience-based Context Optimization

> **Optimizing context for long-horizon coding agents through experience.**

ECO (**Experience-based Context Optimization**) is an experimental framework for improving the performance of long-running software engineering agents by intelligently managing the context they accumulate over time.

Instead of continuously carrying an ever-growing conversation and tool history, ECO explores how an agent can **extract useful engineering experience, retain it, retrieve what matters, and construct optimized context for future reasoning**.

The goal is to make coding agents more reliable, efficient, and capable on complex, long-horizon software engineering tasks.

---

## LongMemEval prototype (ECO V0 research)

Scripts live in the repo root. The LLM and embeddings are always imported from `model.py`.

```bash
python download_dataset.py          # dataset/longmemeval_{s,m,oracle}.json
python create_sample.py             # dataset/longmemeval_sample.json (75 examples)
python inspect_dataset.py --split sample

python run_baselines.py --split sample --limit 5 --dry-run
python run_baselines.py --split sample --limit 5 --judge

python export_samples.py --n 8          # samples/conversation_*.json for research
python run_comparison.py --dry-run        # CSV: full_history vs eco
python run_comparison.py --judge --budgets 2000
```

Baselines: `full_history`, `recent_window`, `rolling_summary`, `refind`, `memorycpt`, `eco`.  
`run_comparison.py` compares **normal agent** (`full_history`) vs **ECO** (`eco`: user-turn retrieval).  
Results are written under `results/` as JSONL/CSV plus a summary JSON.

---

## The Problem

Modern coding agents can autonomously work through complex software engineering tasks involving:

* Hundreds of tool calls
* Large codebases
* Multiple files and dependencies
* Repeated debugging cycles
* Failed and successful approaches
* Architectural discoveries
* Test results
* Changing repository state

However, **the complete trajectory of an agent is not necessarily the best context for its next decision**.

Simply accumulating more history can introduce:

* Context bloat
* Redundant information
* Irrelevant observations
* Repeated reasoning
* Increasing token costs
* Lost attention on important details
* Stale or contradictory information

At the same time, aggressively summarizing history can remove important implementation details and engineering knowledge.

ECO investigates a different approach:

```text
                 Long Agent Trajectory
                         │
                         ▼
              ┌──────────────────────┐
              │         ECO          │
              │                      │
              │ Experience Capture   │
              │ Experience Extraction│
              │ Consolidation        │
              │ Retrieval            │
              │ Relevance Scoring    │
              │ Compression          │
              │ State Tracking       │
              └──────────┬───────────┘
                         │
                         ▼
                  Optimized Context
                         │
                         ▼
                  Coding Agent
```


