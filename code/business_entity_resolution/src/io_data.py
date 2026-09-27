"""Data loading / output writing for the challenge TSV files.

All challenge files are tab-separated. We read with ``dtype=str`` and
``keep_default_na=False`` so that empty fields come through as ``''`` instead
of NaN (the files contain genuinely empty addresses for some pool records).
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from .normalize import process_address, process_name

CHUNK_ROWS = 1_000_000


def load_source(
    path: str,
    limit: int | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Load one source TSV and (optionally) add normalised columns.

    Returns a frame with columns:
    entity_id, country, name_norm, name_k (int8 trailing-suffix count),
    addr_norm, postal
    (raw business_name / business_address and a duplicate "core name" column
    are deliberately NOT kept — memory matters at 12M rows; the core name is
    rebuilt on the fly from norm+k where needed).
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        nrows=limit,
    )
    missing = {"entity_id", "business_name", "business_address", "country"} - set(
        df.columns
    )
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    if not normalize:
        return df

    from .normalize import count_suffix_tokens, norm_text

    norm_names = [norm_text(raw) for raw in df["business_name"].tolist()]
    name_k = np.fromiter(
        (count_suffix_tokens(n) for n in norm_names),
        dtype=np.int8,
        count=len(norm_names),
    )
    addrs = df["business_address"].tolist()
    addr_norms: List[str] = []
    postals: List[str] = []
    for raw in addrs:
        a, p = process_address(raw)
        addr_norms.append(a)
        postals.append(p)

    out = pd.DataFrame(
        {
            "entity_id": df["entity_id"].to_numpy(dtype=object),
            "country": df["country"].to_numpy(dtype=object),
            "name_norm": np.asarray(norm_names, dtype=object),
            "name_k": name_k,
            "addr_norm": np.asarray(addr_norms, dtype=object),
            "postal": np.asarray(postals, dtype=object),
        }
    )
    return out


def load_ground_truth(path: str) -> dict:
    """source1_entity_id -> set of matched S2/S3 ids."""
    gt: dict = {}
    with open(path, encoding="utf-8") as f:
        header = next(f, None)
        if header is None:
            return gt
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1 = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            gt[s1] = {x for x in rest.split(",") if x}
    return gt


def write_pair_file(
    path: str,
    s1_ids: Sequence[str],
    pool_ids: Sequence[str],
    ci: np.ndarray,
    cj: np.ndarray,
    header_name: str = "matched_entity_ids",
) -> Tuple[int, int]:
    """Write a two-column (source1_entity_id, id-list) TSV.

    ``ci``/``cj`` index into ``s1_ids``/``pool_ids``. Every S1 entity gets
    exactly one row; the id list is empty when it has no partners.
    ``ci`` need not be sorted. Returns (rows, non_empty_rows).
    ``header_name`` is ``matched_entity_ids`` for matching_results.tsv and
    ``candidate_entity_ids`` for candidate_pairs.tsv.
    """
    n_s1 = len(s1_ids)
    n_pool = len(pool_ids)
    if len(ci) != len(cj):
        raise ValueError("ci/cj length mismatch")
    if len(ci):
        order = np.argsort(ci, kind="stable")
        ci = ci[order]
        cj = cj[order]
        if int(ci[-1]) >= n_s1 or int(cj.max()) >= n_pool:
            raise ValueError("pair index out of range")

    starts = np.searchsorted(ci, np.arange(n_s1 + 1, dtype=np.int64)) if len(ci) else np.zeros(n_s1 + 1, dtype=np.int64)
    pool_list = list(pool_ids)
    rows = 0
    non_empty = 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", buffering=1 << 22) as f:
        f.write(f"source1_entity_id\t{header_name}\n")
        # chunked row emission keeps the python loop cache-friendly
        for i in range(n_s1):
            seg = cj[starts[i] : starts[i + 1]]
            if seg.shape[0]:
                cell = ",".join([pool_list[j] for j in seg.tolist()])
                non_empty += 1
            else:
                cell = ""
            f.write(s1_ids[i])
            f.write("\t")
            f.write(cell)
            f.write("\n")
            rows += 1
    os.replace(tmp, path)
    return rows, non_empty
