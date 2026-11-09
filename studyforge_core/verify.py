"""Source-grounding verification.

A notes-based item is grounded when its snippet is either an exact normalized
substring of the source, or at least 90% of its tokens appear in one contiguous
source window. Code snippets use whitespace-normalized comparison and
identifier/keyword tokens; code is never executed.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

PROSE_TOKEN_RE = re.compile(r"[a-z0-9_+#.-]+")
CODE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def normalize_prose(text: str) -> str:
    text = _nfkc(text).lower()
    text = re.sub(r"[^\w\s+#.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_code(text: str) -> str:
    text = _nfkc(text).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def prose_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in PROSE_TOKEN_RE.findall(_nfkc(text).lower()):
        token = raw.strip(".-")
        if token:
            tokens.append(token)
    return tokens


def code_tokens(text: str) -> list[str]:
    return CODE_TOKEN_RE.findall(_nfkc(text))


def _window_overlap(needle: list[str], haystack: list[str]) -> float:
    if not needle or not haystack:
        return 0.0
    if len(needle) > len(haystack):
        return 0.0
    size = len(needle)
    best = 0.0
    # A set-based window score is O(n*m) only for large inputs; notes chunks
    # are small (<= ~1,000 words), so this stays cheap.
    needle_counts: dict[str, int] = {}
    for token in needle:
        needle_counts[token] = needle_counts.get(token, 0) + 1
    window_counts: dict[str, int] = {}
    for token in haystack[:size]:
        window_counts[token] = window_counts.get(token, 0) + 1

    def score() -> float:
        matched = sum(min(count, window_counts.get(token, 0)) for token, count in needle_counts.items())
        return matched / size

    best = score()
    for index in range(size, len(haystack)):
        outgoing = haystack[index - size]
        incoming = haystack[index]
        window_counts[outgoing] -= 1
        if window_counts[outgoing] == 0:
            del window_counts[outgoing]
        window_counts[incoming] = window_counts.get(incoming, 0) + 1
        current = score()
        if current > best:
            best = current
            if best >= 0.99:
                break
    return best


def verify_snippet(snippet: str | None, source: str, *, code: bool = False) -> dict[str, Any]:
    if not snippet or not str(snippet).strip() or not source:
        return {"grounded": False, "match": "none", "score": 0.0}
    snippet = str(snippet)
    if code:
        needle = normalize_code(snippet)
        haystack = normalize_code(source)
        tokens = code_tokens(snippet)
    else:
        needle = normalize_prose(snippet)
        haystack = normalize_prose(source)
        tokens = prose_tokens(snippet)

    if needle and needle in haystack:
        return {"grounded": True, "match": "exact", "score": 1.0}
    if code:
        # Code must match after whitespace normalization. A wrong operator or
        # identifier is not "90% grounded"; it is a different program.
        return {"grounded": False, "match": "none", "score": 0.0}
    if len(tokens) < 3:
        return {"grounded": False, "match": "none", "score": 0.0}
    haystack_tokens = code_tokens(source) if code else prose_tokens(source)
    overlap = _window_overlap(tokens, haystack_tokens)
    return {
        "grounded": overlap >= 0.9,
        "match": "fuzzy" if overlap >= 0.9 else "none",
        "score": round(overlap, 4),
    }
