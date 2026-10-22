"""Registry of context-optimization systems compared in this repo. Every system implements
build_memory(q) -> memory and build_context(query, memory, token_budget, query_date) -> ContextResult."""
from __future__ import annotations

from core.context import Services

# Span-level ECO variants: identical everywhere except the objective flags (systems/eco/objective.py).
SPAN_VARIANTS = {
    "eco_spans": ("ECO-spans", {}),                                    # plain facility location over spans
    "eco_sup": ("ECO-spans+supersede", {"supersede": 0.5}),            # old restatements cover the new one halfway
    "eco_clos": ("ECO-spans+one-per-session", {"one_per": "session"}),  # hard breadth cap, one span per session
    "eco_facet": ("ECO-spans+facets", {"facets": 3}),                  # cover 3 informative query terms
    "eco_full": ("ECO-spans+all", {"supersede": 0.5, "one_per": "session", "facets": 3}),
    "eco_sess": ("ECO-spans+soft-breadth", {"session_weight": 1.0}),   # breadth worth as much as relevance
    "eco_sess_x2": ("ECO-spans+soft-breadth-x2", {"session_weight": 2.0}),  # dose-response probe
    "eco_all": ("ECO-spans+supersede+soft-breadth+facets", {"supersede": 0.5, "session_weight": 1.0, "facets": 3}),
}

LLM_FREE = {"recent", "bm25", "dense", "hybrid", "mmr", "oracle", "eco",
            "span_hybrid", "span_mmr"} | set(SPAN_VARIANTS)
NEEDS_EMBEDDINGS = {"dense", "hybrid", "mmr", "oracle", "eco", "eco_facts", "memorycpt",
                    "span_hybrid", "span_mmr"} | set(SPAN_VARIANTS)


def build_system(name: str, services: Services, construction_parallel: int = 4):
    from systems.baselines.simple import BM25Turns, DenseTurns, HybridRRF, MMRTurns, OracleEvidence, RecentWindow
    from systems.baselines.span import SpanHybridRRF, SpanMMR
    from systems.eco.method import ECO
    from systems.eco.spans import ECOspans
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
        "span_hybrid": lambda: SpanHybridRRF(services),
        "span_mmr": lambda: SpanMMR(services),
        **{key: (lambda k=key: ECOspans(services, name=SPAN_VARIANTS[k][0], **SPAN_VARIANTS[k][1]))
           for key in SPAN_VARIANTS},
        "eco_facts": lambda: ECO(services, use_facts=True, construction_parallel=construction_parallel),
        "refind": lambda: ReFind(services),
        "memorycpt": lambda: MemoryCPTLite(services, construction_parallel=construction_parallel),
    }
    if name not in factories:
        raise KeyError(f"unknown system {name!r}; known: {sorted(factories)}")
    return factories[name]()
