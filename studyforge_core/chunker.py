"""Code-aware notes chunker.

Rules:
* never split inside a fenced code block;
* keep a code block with the paragraph that introduces it;
* prefer heading boundaries, then blank-line blocks, targeting 600-900 words;
* stable IDs for the same pack source and order.
"""

from __future__ import annotations

import hashlib
import re

from .models import Chunk, word_count

FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_notes(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    lines = [line.rstrip() for line in text.split("\n")]
    out: list[str] = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            if blank_run <= 2:
                out.append("")
        else:
            blank_run = 0
            out.append(line)
    return "\n".join(out).strip()


def split_blocks(text: str) -> list[str]:
    """Split on blank lines while treating fenced code as one unit.

    When a fence starts after a paragraph, the paragraph is pulled into the
    same block so the code keeps its explanation.
    """
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0

    def flush() -> None:
        nonlocal current
        if current:
            block = "\n".join(current).strip("\n")
            if block.strip():
                blocks.append(block)
            current = []

    for line in text.split("\n"):
        match = FENCE_RE.match(line)
        if match:
            marker = match.group(2)
            if not in_fence:
                if not current and blocks:
                    previous = blocks[-1]
                    if word_count(previous) <= 160 or is_heading(previous):
                        current = blocks.pop().split("\n")
                current.append(line)
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
                continue
            if marker[0] == fence_char and len(marker) >= fence_len:
                current.append(line)
                in_fence = False
                fence_char = ""
                fence_len = 0
                flush()
                continue
        if in_fence:
            current.append(line)
        elif not line.strip():
            flush()
        else:
            current.append(line)
    flush()
    return blocks


def is_code_block(block: str) -> bool:
    return any(FENCE_RE.match(line) for line in block.split("\n"))


def is_heading(block: str) -> bool:
    first = next((line for line in block.split("\n") if line.strip()), "")
    return bool(HEADING_RE.match(first))


def heading_text(block: str) -> str:
    first = next((line for line in block.split("\n") if line.strip()), "")
    match = HEADING_RE.match(first)
    return match.group(2).strip() if match else ""


def _first_heading(block: str) -> str:
    for line in block.split("\n"):
        match = HEADING_RE.match(line)
        if match:
            return match.group(2).strip()
    return ""


def split_long_block(block: str, max_words: int) -> list[str]:
    """Split prose on sentences; never split a fenced code block."""
    if is_code_block(block):
        return [block]
    sentences = [s for s in SENTENCE_RE.split(block) if s.strip()]
    pieces: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = word_count(sentence)
        if sentence_words > max_words:
            if current:
                pieces.append(" ".join(current).strip())
                current = []
                current_words = 0
            words = sentence.split()
            for start in range(0, len(words), max_words):
                pieces.append(" ".join(words[start : start + max_words]))
            continue
        if current and current_words + sentence_words > max_words:
            pieces.append(" ".join(current).strip())
            current = []
            current_words = 0
        current.append(sentence.strip())
        current_words += sentence_words
    if current:
        pieces.append(" ".join(current).strip())
    return [p for p in pieces if p.strip()]


def chunk_notes(
    notes: str,
    pack_id: str,
    min_words: int = 600,
    max_words: int = 900,
) -> list[Chunk]:
    text = normalize_notes(notes)
    if not text:
        return []

    pieces: list[tuple[str, str]] = []
    for block in split_blocks(text):
        if is_code_block(block):
            pieces.append(("code", block))
        elif is_heading(block):
            pieces.append(("heading", block))
        else:
            for piece in split_long_block(block, max_words):
                pieces.append(("prose", piece))

    chunks: list[tuple[str, str]] = []
    current: list[str] = []
    current_words = 0
    section_heading = "Notes"

    def flush() -> None:
        nonlocal current, current_words
        if not current:
            return
        body = "\n\n".join(current).strip()
        if not body:
            current = []
            current_words = 0
            return
        heading = _first_heading(body) or section_heading
        chunks.append((body, heading))
        current = []
        current_words = 0

    for kind, piece in pieces:
        words = word_count(piece)
        if kind == "heading":
            if current_words >= 400:
                flush()
            section_heading = heading_text(piece) or section_heading
            current.append(piece)
            current_words += words
            continue

        if kind == "code":
            if current and current_words + words > max_words and current_words >= 300:
                flush()
            current.append(piece)
            current_words += words
            if current_words >= max_words:
                flush()
            continue

        if current and current_words + words > max_words and current_words >= min_words:
            flush()
        if current_words + words > max_words and current_words == 0 and words > max_words:
            for sub in split_long_block(piece, max_words):
                current.append(sub)
                current_words += word_count(sub)
                if current_words >= max_words:
                    flush()
        else:
            current.append(piece)
            current_words += words

    flush()

    if len(chunks) > 1 and word_count(chunks[-1][0]) < 80:
        last = chunks.pop()
        prev_body, prev_heading = chunks[-1]
        merged = prev_body + "\n\n" + last[0]
        if word_count(merged) <= 1100:
            chunks[-1] = (merged, prev_heading)
        else:
            chunks.append(last)

    out: list[Chunk] = []
    suffix = pack_id[-8:]
    for index, (body, heading) in enumerate(chunks):
        out.append(
            Chunk(
                id=f"chunk_{suffix}_{index:03d}",
                pack_id=pack_id,
                index=index,
                text=body,
                heading=heading,
                text_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                word_count=word_count(body),
            )
        )
    return out


def chunk_topic(topic: str, pack_id: str) -> list[Chunk]:
    body = f"Topic: {topic.strip()}"
    suffix = pack_id[-8:]
    return [
        Chunk(
            id=f"chunk_{suffix}_000",
            pack_id=pack_id,
            index=0,
            text=body,
            heading="Topic outline",
            text_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            word_count=word_count(body),
        )
    ]
