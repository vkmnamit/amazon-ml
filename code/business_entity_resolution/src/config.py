"""Shared configuration for the entity-resolution pipeline."""

from dataclasses import dataclass


@dataclass
class BlockingConfig:
    """Knobs for candidate generation.

    max_pool_per_key: drop a blocking key when it appears on more than this
        many pool (S2/S3) records — very common keys only add noise.
    max_prod_per_key: drop a key when n_S1(key) * n_pool(key) exceeds this —
        bounds the number of raw pairs a single key can contribute.
    top_k: maximum candidates retained per Source-1 entity (ranked by number
        of distinct blocking keys shared).
    """

    max_pool_per_key: int = 400
    max_prod_per_key: int = 20_000
    top_k: int = 60


@dataclass
class ScoringConfig:
    # Default accept threshold for the heuristic scorer (tuned for precision /
    # F_0.5). The `train` command overwrites this with a tuned value in
    # artifacts/threshold.json when the training sources are available.
    heuristic_threshold: float = 0.58
