def evidence_recall(source_turn_ids, evidence_turn_ids) -> float | None:
    """Fraction of gold evidence turns included in the context; None if the question has no evidence."""
    if not evidence_turn_ids:
        return None
    s = set(source_turn_ids)
    return sum(e in s for e in evidence_turn_ids) / len(evidence_turn_ids)
