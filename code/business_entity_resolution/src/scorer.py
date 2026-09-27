"""Pair scorers: trained model (when available) or heuristic fallback."""

from __future__ import annotations

import json
import os
import pickle
from typing import Optional

import numpy as np

from .config import ScoringConfig
from .features import N_FEATURES

MODEL_FEATURE_INDEX = {name: i for i, name in enumerate(
    [
        "name_token_set_core",
        "name_token_sort",
        "name_ratio",
        "addr_token_sort",
        "addr_ratio",
        "addr_both_present",
        "postal_match",
    ]
)}


def heuristic_scores(X: np.ndarray) -> np.ndarray:
    """Precision-leaning hand-crafted score in [0, 1].

    Used when no trained model is available (the training source files are
    absent from the workspace). Weights were chosen so that:
      * a near-exact name match with a corroborating address passes easily,
      * a strong name with no address at all sits just above the threshold,
      * weak/absent names never pass regardless of address.
    """
    name_best = np.maximum(X[:, 0], X[:, 1])
    addr_best = np.maximum(X[:, 3], X[:, 4]) * X[:, 5]
    s = (
        0.50 * X[:, 0]  # core token-set overlap (suffix-invariant)
        + 0.18 * X[:, 1]  # order-invariant full-name similarity
        + 0.08 * X[:, 2]  # strict name similarity
        + 0.26 * addr_best  # address similarity (0 when either missing)
        + 0.14 * X[:, 6]  # postal-code match bonus
    )
    return np.minimum(s, 1.0)


class Scorer:
    """Scores candidate pairs; wraps either a pickled sklearn model or the
    heuristic above."""

    def __init__(self, model=None, threshold: float | None = None,
                 cfg: Optional[ScoringConfig] = None):
        cfg = cfg or ScoringConfig()
        self.model = model
        self.threshold = (
            threshold if threshold is not None else cfg.heuristic_threshold
        )

    @property
    def kind(self) -> str:
        return "model" if self.model is not None else "heuristic"

    def score_block(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.empty(0, dtype=np.float32)
        if self.model is not None:
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)[:, 1].astype(np.float32)
            return self.model.predict(X).astype(np.float32)
        return heuristic_scores(X).astype(np.float32)

    @staticmethod
    def load(model_path: str, threshold: float | None = None,
             cfg: Optional[ScoringConfig] = None) -> "Scorer":
        model = None
        resolved_threshold = threshold
        if model_path and os.path.isfile(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        thr_path = None
        if model_path:
            cand = os.path.join(os.path.dirname(model_path), "threshold.json")
            if os.path.isfile(cand):
                thr_path = cand
        if resolved_threshold is None and thr_path:
            with open(thr_path, encoding="utf-8") as f:
                resolved_threshold = float(json.load(f)["threshold"])
        return Scorer(model=model, threshold=resolved_threshold, cfg=cfg)


def save_model(model, threshold: float, model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(
        os.path.join(os.path.dirname(model_path) or ".", "threshold.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump({"threshold": float(threshold), "n_features": N_FEATURES}, f)
