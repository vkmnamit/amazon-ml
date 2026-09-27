"""Scoring metrics: macro-averaged F_0.5 (the challenge metric)."""

from __future__ import annotations

from typing import Dict, Iterable, Set

BETA = 0.5
BETA2 = BETA * BETA


def f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def entity_f05(predicted: Set[str], truth: Set[str]) -> float:
    """F_0.5 for a single Source-1 entity.

    - both empty (correct singleton) -> 1.0
    - truth empty, prediction non-empty -> 0.0 (false merge)
    - truth non-empty, prediction empty -> 0.0
    """
    if not truth and not predicted:
        return 1.0
    if not truth or not predicted:
        return 0.0
    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return f_beta(precision, recall)


def macro_f05(
    predictions: Dict[str, Iterable[str]], truth: Dict[str, Iterable[str]]
) -> float:
    """Macro F_0.5 over every entity present in ``truth``."""
    if not truth:
        return 0.0
    total = 0.0
    for s1, gt in truth.items():
        gt_set = set(gt)
        pred_set = set(predictions.get(s1, ()))
        total += entity_f05(pred_set, gt_set)
    return total / len(truth)


def parse_id_list(cell: str) -> Set[str]:
    """'S2-1,S3-2' -> {'S2-1','S3-2'}; empty cell -> set()."""
    cell = (cell or "").strip()
    if not cell:
        return set()
    return {p.strip() for p in cell.split(",") if p.strip()}
