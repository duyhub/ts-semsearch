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
from pathlib import Path
from typing import Iterable, Sequence

# Abbreviation / slang dictionary — keys and values are FOLDED (see fold()).
# Query-side only; documents are already canonical. Seeded from SPEC §3 + the
# dataset's cities/categories; extend from eval failures.
ABBREVIATIONS: dict[str, str] = {
    "hcm": "tp hcm", "sg": "tp hcm", "tphcm": "tp hcm", "hcmc": "tp hcm", "sai gon": "tp hcm",
    "hn": "ha noi", "dn": "da nang", "dl": "da lat",
    "cf": "ca phe", "cafe": "ca phe", "cofe": "ca phe", "coffee": "ca phe", "quan cf": "quan ca phe",
    "ks": "khach san", "hotel": "khach san",
    "nh": "nha hang", "restaurant": "nha hang",
    "tt": "trung tam", "tttm": "trung tam thuong mai", "mall": "trung tam thuong mai",
    "bv": "benh vien", "hospital": "benh vien",
    "atm": "atm", "gas": "tram xang", "gas station": "tram xang",
}

_Q_RE = re.compile(r"^q(\d{1,2})$")  # q1..q12 -> "quan N"

# Folded connectives + generic quality adjectives + generic place words. Used to
# strip filler when detecting whether a query is "fully explained" (SPEC §6 hard
# filters). MUST NOT contain category/subject nouns (bún, chả, mua, sắm, ...) — a
# leftover subject noun is exactly what blocks a category hard-filter (guards P019/P055).
STOPWORDS: frozenset[str] = frozenset({
    # connectives / prepositions / determiners
    "o", "cho", "co", "cua", "va", "voi", "de", "tai", "gan", "day", "nay", "do",
    "ra", "vao", "len", "xuong", "khu", "vuc", "khong", "qua", "cung", "hay", "hoac",
    "mot", "cac", "nhung", "vai", "moi",
    # generic place / intent words
    "noi", "cho", "dia", "diem", "quan", "tim", "kiem", "muon", "can", "gi", "nao",
    "phu", "hop", "the",
    # generic quality adjectives
    "ngon", "dep", "re", "tot", "sang", "chanh", "xin", "cu", "lon", "nho", "rong",
    "rai", "sach", "se", "gia", "noi tieng", "tieng",
    # generic descriptors (incl. common English) that can appear in POI names but are
    # not subjects: view/city/scenery, dining verbs, time words
    "view", "thanh", "an", "uong", "mon", "gio",
})


def contains_token_seq(haystack_folded: str, key_folded: str) -> bool:
    """True iff `key_folded` appears as a contiguous token subsequence of
    `haystack_folded` (both already folded). Token-boundary match, so the district
    key "quan 1" does NOT match inside "quan 10"/"quan 12" (the substring collision)."""
    ht = haystack_folded.split()
    kt = key_folded.split()
    m = len(kt)
    if m == 0:
        return False
    return any(ht[i:i + m] == kt for i in range(len(ht) - m + 1))


def _display_tokens(raw: str) -> list[str]:
    """Diacritic-preserving tokens aligned 1:1 with ``fold(raw).split()``.

    A character joins the current token iff it survives folding (letters, digits,
    '/'); the string is NFC-normalized + lowercased first so each surviving display
    char folds to exactly one folded char — guaranteeing ``display[i]`` folds to the
    i-th folded token and that the two are equal length. Stray combining marks
    attach to the current token (they fold away). This lets a matcher align the raw
    query's diacritics against a key's display form character by character.
    """
    s = unicodedata.normalize("NFC", raw).lower()
    tokens: list[str] = []
    cur: list[str] = []
    for ch in s:
        if unicodedata.category(ch) == "Mn":  # combining mark: keep, folds to nothing
            if cur:
                cur.append(ch)
            continue
        if fold(ch):  # survives folding -> part of a token
            cur.append(ch)
        elif cur:  # separator -> flush
            tokens.append("".join(cur))
            cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens


def _char_compatible(q_char: str, k_char: str) -> bool:
    """A single aligned char is diacritic-compatible if the query char is plain
    (equals its own folded form) or exactly matches the key's display char. A
    diacritic-bearing query char must not contradict the key's char."""
    return fold(q_char) == q_char or q_char == k_char


def _span_compatible(raw_span: Sequence[str], key_span: Sequence[str]) -> bool:
    for rt, kt in zip(raw_span, key_span):
        if len(rt) != len(kt) or any(
            not _char_compatible(rc, kc) for rc, kc in zip(rt, kt)
        ):
            return False
    return True


def compat_token_seq(raw_text: str, key_display: str) -> bool:
    """True iff ``key_display``'s folded form appears as a contiguous token
    subsequence of ``fold(raw_text)`` AND at least one such occurrence is
    diacritic-COMPATIBLE with ``key_display`` (Fix 1).

    Unaccented query input stays permissive — folding exists precisely so plain
    input matches — but any diacritic the query DOES type must agree with the key:
    'phở có' does NOT match key 'Phố Cổ' (ở≠ố) while 'pho co' and 'phố cổ' both do;
    'sáng' does NOT match key 'sang' though plain 'sang' does. Token-boundary
    matching also stops 'park' firing inside 'parking' or 'late' inside 'chocolate'.
    """
    key_fold = fold(key_display).split()
    m = len(key_fold)
    if m == 0:
        return False
    raw_disp = _display_tokens(raw_text)
    raw_fold = [fold(t) for t in raw_disp]
    key_disp = _display_tokens(key_display)
    if len(key_disp) != m:  # defensive: key must tokenize to its folded length
        key_disp = key_fold
    for i in range(len(raw_fold) - m + 1):
        if raw_fold[i:i + m] == key_fold and _span_compatible(raw_disp[i:i + m], key_disp):
            return True
    return False


def token_key_matches(hay_folded: str, raw_text: str, key_display: str) -> bool:
    """Keyword match against a (possibly abbreviation-expanded) folded haystack,
    gated by diacritic compatibility with the RAW query (Fix 1).

    The key must appear as a token subsequence of ``hay_folded`` — so 'q1' -> 'quan 1'
    and 'cf' -> 'ca phe' expansions still resolve. If the key ALSO appears literally
    in the raw query it must be diacritic-compatible there (rejects 'sáng' vs key
    'sang'); if it only arose via expansion there is no raw diacritic to contradict,
    so the match stands.
    """
    key_fold = fold(key_display)
    if not contains_token_seq(hay_folded, key_fold):
        return False
    if not contains_token_seq(fold(raw_text), key_fold):
        return True  # arrived via expansion; nothing in the raw query to contradict
    return compat_token_seq(raw_text, key_display)


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


_RESOURCES = Path(__file__).resolve().parent / "resources"


@lru_cache(maxsize=1)
def _vi_common_by_fold() -> dict[str, tuple[str, ...]]:
    """Map a folded token to the diacritic display forms of the Vietnamese common
    words that fold to it (Fix 2). Vendored from stopwords-iso/stopwords-vi (MIT);
    see resources/vi_common_words.txt. Diacritic-aware storage is what lets the
    plain common word 'cha' (father/negation) coexist with the food subject 'chả'
    without one masking the other."""
    by_fold: dict[str, list[str]] = {}
    with open(_RESOURCES / "vi_common_words.txt", encoding="utf-8") as fh:
        for line in fh:
            word = line.strip()
            if not word or word.startswith("#"):
                continue
            key = fold(word)
            if key:
                by_fold.setdefault(key, []).append(unicodedata.normalize("NFC", word).lower())
    return {k: tuple(v) for k, v in by_fold.items()}


def query_common_tokens(text: str) -> set[str]:
    """Folded query tokens that are Vietnamese common words (the vendored stopword
    resource), matched DIACRITIC-COMPATIBLY against the raw query (Fix 2).

    A flagged token can never be a distinctive subject and is dropped from the
    parser's residual. Diacritic-aware: the superlative particle 'nhất' is flagged
    (accented or plain 'nhat'), but the food subject 'chả' is NOT flagged by the
    unrelated common word 'cha'. Only checks tokens that actually occur in the
    query, so it stays cheap.
    """
    by_fold = _vi_common_by_fold()
    out: set[str] = set()
    for tok in set(fold(text).split()):
        for disp in by_fold.get(tok, ()):
            if compat_token_seq(text, disp):
                out.add(tok)
                break
    return out
