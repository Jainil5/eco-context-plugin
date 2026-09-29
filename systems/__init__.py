"""Registry of context-optimization systems compared in this repo. Every system implements
build_memory(q) -> memory and build_context(query, memory, token_budget, query_date) -> ContextResult."""
from __future__ import annotations

from core.context import Services

LLM_FREE = {"recent", "bm25", "dense", "hybrid", "mmr", "oracle", "eco"}  # no chat-model calls (eco uses embeddings)
NEEDS_EMBEDDINGS = {"dense", "hybrid", "mmr", "oracle", "eco", "eco_facts", "memorycpt"}


def build_system(name: str, services: Services, construction_parallel: int = 4):
    from systems.baselines.simple import BM25Turns, DenseTurns, HybridRRF, MMRTurns, OracleEvidence, RecentWindow
    from systems.eco.method import ECO
    from systems.memorycpt.memorycpt import MemoryCPTLite
    from systems.refind.refind import ReFind
    factories = {
        "recent": lambda: RecentWindow(),
        "bm25": lambda: BM25Turns(),
        "dense": lambda: DenseTurns(services),
        "hybrid": lambda: HybridRRF(services),
        "mmr": lambda: MMRTurns(services),
        "oracle": lambda: OracleEvidence(HybridRRF(services)),  # upper bound: gold turns first
        "eco": lambda: ECO(services, use_facts=False),
        "eco_facts": lambda: ECO(services, use_facts=True, construction_parallel=construction_parallel),
        "refind": lambda: ReFind(services),
        "memorycpt": lambda: MemoryCPTLite(services, construction_parallel=construction_parallel),
    }
    if name not in factories:
        raise KeyError(f"unknown system {name!r}; known: {sorted(factories)}")
    return factories[name]()
