"""Text / address normalisation for the entity-resolution pipeline.

Everything here is pure stdlib (re + unicodedata) and is applied row-wise to
the ~12M records of the challenge files, so the functions are written to be
cheap: one NFKC pass, one regex substitution pass, one whitespace collapse.

Design notes
------------
* ``norm_text``      -> canonical surface form used for fuzzy comparison.
* ``core_name``      -> ``norm_text`` with leading articles and trailing legal
                        suffixes (Inc, Pvt, SARL, ...) stripped, used for the
                        "core" similarity feature and for blocking keys.
* ``process_addr``   -> normalised address + extracted postal code. Postal
                        extraction is deliberately country-agnostic (the test
                        set introduces France, which never appears in
                        training) so the pipeline stays an open set w.r.t.
                        country labels.
"""

from __future__ import annotations

import re
import unicodedata

# Tokens that only ever appear as legal-entity suffixes. Stripped from the end
# of the name (repeatedly) to obtain the comparable "core" name.
LEGAL_SUFFIXES = frozenset(
    {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "llc",
        "llp",
        "lp",
        "ltd",
        "limited",
        "pvt",
        "private",
        "co",
        "company",
        "plc",
        "gmbh",
        "srl",
        "sarl",
        "sa",
        "sas",
        "sasu",
        "eurl",
        "sas",
        "bv",
        "nv",
        "ag",
        "oy",
        "ab",
        "pty",
        "pte",
        "srls",
        "snc",
        "sci",
    }
)

LEADING_ARTICLES = frozenset({"the"})

# Word-level address abbreviations -> expansion. Applied only to addresses.
# Only reasonably unambiguous pairs are listed; ambiguous ones (e.g. "st"
# could be *Saint*) are left alone rather than risking corrupt keys.
ADDRESS_EXPANSIONS = {
    # US / UK style
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "pkwy": "parkway",
    "cir": "circle",
    "ct": "court",
    "sq": "square",
    "ter": "terrace",
    "pl": "place",
    "expy": "expressway",
    "fwy": "freeway",
    "trl": "trail",
    "pt": "point",
    "fl": "floor",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "hwy": "highway",
    "rt": "route",
    "usrte": "route",
    # French style
    "bd": "boulevard",
    "boul": "boulevard",
    "boulvd": "boulevard",
    "imp": "impasse",
    "qu": "quai",
    "all": "allee",
    "che": "chemin",
    "chaus": "chaussee",
    # India style
    "nr": "near",
    "opp": "opposite",
    "beside": "near",
    "hno": "house",
    "hno house": "house",
}

# Keep letters + digits of ANY script, everything else becomes a space.
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_WS = re.compile(r"\s+")

# Postal-code digits. Strategy: find all runs of 5-6 digits; prefer the last
# 6-digit run (Indian PIN codes) else the last 5-digit run (US ZIP / French
# code). Positional preference for "last" matches how postal codes usually sit
# at the end of an address in all three countries.
_POSTAL_RUN = re.compile(r"(?<!\d)\d{5,6}(?!\d)")


def norm_text(raw: str) -> str:
    """Lowercase, NFKC-fold, '&' -> 'and', strip punctuation, collapse spaces."""
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", raw).lower()
    s = s.replace("&", " and ")
    s = _NON_WORD.sub(" ", s)
    s = s.replace("_", " ")
    return _WS.sub(" ", s).strip()


def core_name(norm: str) -> str:
    """Strip leading article and trailing legal suffixes from a normalised name."""
    if not norm:
        return ""
    toks = norm.split()
    while toks and toks[0] in LEADING_ARTICLES:
        toks = toks[1:]
    while toks and toks[-1] in LEGAL_SUFFIXES:
        toks = toks[:-1]
    return " ".join(toks)


def process_name(raw: str) -> tuple[str, str]:
    """Return (norm_name, core_name)."""
    n = norm_text(raw)
    return n, core_name(n)


def count_suffix_tokens(norm: str) -> int:
    """Number of trailing legal-suffix tokens in a normalised name.

    Stored per record as int8 instead of materialising a second "core name"
    string column (memory matters at 12M rows). -1 means the whole name is
    legal suffixes (core is empty). Used to rebuild the core name for the
    token-set feature inside the feature workers.
    """
    if not norm:
        return -1
    toks = norm.split()
    n = 0
    while n < len(toks) and toks[-1 - n] in LEGAL_SUFFIXES:
        n += 1
    if n == len(toks):
        return -1
    return n


def norm_address(raw: str) -> str:
    """Normalised address with word-level abbreviations expanded."""
    n = norm_text(raw)
    if not n:
        return ""
    toks = [ADDRESS_EXPANSIONS.get(t, t) for t in n.split()]
    return " ".join(toks)


def extract_postal(norm_addr: str) -> str:
    """Best-effort postal code from a normalised address ('' when absent)."""
    if not norm_addr:
        return ""
    last5 = ""
    last6 = ""
    for m in _POSTAL_RUN.finditer(norm_addr):
        v = m.group(0)
        if len(v) == 6:
            last6 = v
        else:
            last5 = v
    return last6 or last5


def process_address(raw: str) -> tuple[str, str]:
    """Return (norm_address, postal_code)."""
    a = norm_address(raw)
    return a, extract_postal(a)


_VOWELS = re.compile(r"[aeiouy]+")


def name_skeleton(token: str) -> str:
    """Consonant skeleton of a token ('zephay' -> 'zphy') — typo/variant key."""
    return _VOWELS.sub("", token)
