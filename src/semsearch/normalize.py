"""Vietnamese text normalization (SPEC §3).

Folding + tokenization happen only inside indexes/matchers; raw diacritics are
preserved for display (NFR-4). Query-side abbreviation/slang expansion and a
single fuzzy `canonicalize()` primitive (the C2 consolidation — one matcher, not
three) map degraded input onto the closed vocabularies before retrieval.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable, Sequence

# Abbreviation / slang dictionary — keys and values are FOLDED (see fold()).
# Query-side only; documents are already canonical. Seeded from SPEC §3 + the
# dataset's cities/categories; extend from eval failures.
ABBREVIATIONS: dict[str, str] = {
    "hcm": "tp hcm", "sg": "tp hcm", "tphcm": "tp hcm", "hcmc": "tp hcm",
    "hn": "ha noi", "dn": "da nang", "dl": "da lat",
    "cf": "ca phe", "cafe": "ca phe", "cofe": "ca phe", "coffee": "ca phe", "quan cf": "quan ca phe",
    "ks": "khach san", "hotel": "khach san",
    "nh": "nha hang", "restaurant": "nha hang",
    "tt": "trung tam", "tttm": "trung tam thuong mai", "mall": "trung tam thuong mai",
    "bv": "benh vien", "hospital": "benh vien",
    "atm": "atm", "gas": "tram xang", "gas station": "tram xang",
}

_Q_RE = re.compile(r"^q(\d{1,2})$")  # q1..q12 -> "quan N"


def fold(s: str) -> str:
    """NFD -> strip combining marks, đ->d, lowercase, keep alnum + '/', collapse space."""
    if not s:
        return ""
    s = s.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    # keep letters/digits and '/' (for "24/7"); everything else -> space
    s = re.sub(r"[^a-z0-9/]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(s: str) -> list[str]:
    """Fold then split into tokens. '24/7' is preserved as a single token."""
    folded = fold(s)
    return [t for t in folded.split(" ") if t]


def _levenshtein_le1(a: str, b: str) -> bool:
    """True iff edit distance(a, b) <= 1. Cheap early-outs; no full DP needed."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:  # single substitution
        return sum(x != y for x, y in zip(a, b)) == 1
    # one insertion/deletion: walk the shorter against the longer
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    edited = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            if edited:
                return False
            edited = True
            j += 1
    return True


def canonicalize(token: str, vocab: Sequence[str], *, max_edit: int = 1, min_len: int = 4) -> str | None:
    """Single fuzzy-match primitive (C2). Return the canonical vocab term for
    `token`, or None if no confident match.

    Exact match wins. Fuzzy (edit distance <= max_edit) only fires for tokens of
    length >= min_len, and ONLY when the match is UNAMBIGUOUS — if the token is
    within max_edit of two different vocab terms, we refuse (return None) rather
    than risk mis-mapping a real word onto an unrelated one (the C2 precision
    guard). `token` and `vocab` are expected already folded.
    """
    if token in vocab:
        return token
    if max_edit < 1 or len(token) < min_len:
        return None
    hits = [v for v in vocab if len(v) >= min_len and _levenshtein_le1(token, v)]
    return hits[0] if len(hits) == 1 else None


def expand_query(text: str) -> list[str]:
    """Query normalization for the lexical (BM25) side: fold, expand
    abbreviations/slang and q-numbers, return tokens. Multi-word abbreviations
    are matched greedily on bigrams before unigrams.
    """
    tokens = tokenize(text)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        # bigram abbreviation (e.g. "gas station", "quan cf")
        if i + 1 < len(tokens):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            if bigram in ABBREVIATIONS:
                out.extend(ABBREVIATIONS[bigram].split())
                i += 2
                continue
        tok = tokens[i]
        m = _Q_RE.match(tok)
        if m:
            out.extend(["quan", m.group(1)])
        elif tok in ABBREVIATIONS:
            out.extend(ABBREVIATIONS[tok].split())
        else:
            out.append(tok)
        i += 1
    return out


@lru_cache(maxsize=2048)
def _fold_cached(s: str) -> str:
    return fold(s)


def doc_tokens(text: str) -> list[str]:
    """Tokenization for documents — fold only (docs are canonical, no expansion)."""
    return [t for t in _fold_cached(text).split(" ") if t]
