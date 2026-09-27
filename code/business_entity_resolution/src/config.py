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

    max_pool_per_key: int = 2000
    max_prod_per_key: int = 200_000
    top_k: int = 120
    # Emit the low-information "coverage" key families (unigrams, token pairs,
    # token windows). Measured on a 20k-entity train sample: ON costs 2.8x the
    # candidates for +0.04 pair recall; at an equal per-entity budget OFF wins
    # (0.734 @ 38 cands/S1 vs 0.720 @ 58). Default OFF — candidate volume is
    # only bounded by scoring time, and the pairs it adds are near-threshold
    # junk the model has to reject anyway.
    weak_keys: bool = False


@dataclass
class ScoringConfig:
    # Default accept threshold for the heuristic scorer (tuned for precision /
    # F_0.5). The `train` command overwrites this with a tuned value in
    # artifacts/threshold.json when the training sources are available.
    heuristic_threshold: float = 0.58
