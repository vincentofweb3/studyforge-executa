"""Prompt contracts for host sampling.

Every prompt asks for JSON only. The parser in :mod:`llm` is defensive anyway.
Source snippets must be copied verbatim; the verifier in :mod:`verify` is the
final authority and relabels or drops ungrounded items.
"""

from __future__ import annotations

import json
from typing import Any

EXTRACT_SYSTEM = """You build study packs for technical learners.
Return JSON only, with no markdown fences or commentary.
Extract 4-10 concepts from the supplied notes. Use only facts in the notes.
Each concept has:
- name: 2-6 words, specific and unique
- definition: one clear sentence
- difficulty: integer 1-5
- prerequisites: array of concept names that must be learned first (may be empty)
- track: one of databases, networking, algorithms, operating_systems,
  system_design, cloud_devops, security, languages, other
- source_snippet: a verbatim span from the notes (10-40 words) that supports
  the definition. Copy it exactly; do not paraphrase or use ellipses.
If the notes are too thin, return fewer concepts. Never invent facts."""

TOPIC_SYSTEM = """You build study packs for technical learners.
Return JSON only, with no markdown fences or commentary.
Create a concept outline for the requested topic at a beginner-to-intermediate
level. Return 8-15 concepts. Use only well-established technical knowledge.
Each concept has:
- name: 2-6 words, specific and unique
- definition: one clear sentence
- difficulty: integer 1-5
- prerequisites: array of concept names that must be learned first (may be empty)
- track: one of databases, networking, algorithms, operating_systems,
  system_design, cloud_devops, security, languages, other
There are no notes in topic mode, so omit source_snippet. Every generated item
will be labelled General knowledge."""

ITEMS_SYSTEM = """You build grounded flashcards and quiz items for technical learners.
Return JSON only, with no markdown fences or commentary.
For each concept, write 2-4 cards and 1-2 quiz items.
Cards:
- concept_id and concept_name
- kind: definition | compare | cloze | complexity | predict_output | spot_bug
- front, back
- code and language only when the card includes a code snippet
- source_snippet: a verbatim span from the supplied notes. Copy it exactly.
- not_executed: true for predict_output/spot_bug cards; code is never executed.
Quiz:
- concept_id and concept_name
- kind: mcq | short
- prompt; options (3-4) and answer for mcq; rubric (1-4 points) for short
- explanation tied to the source snippet
- source_snippet: a verbatim span from the supplied notes
Rules:
- In notes mode, every item must be supported by the notes. If you cannot find
  a verbatim snippet, set grounded:false and omit the snippet; the item will be
  labelled General knowledge.
- MCQ distractors must be plausible mistakes, not jokes. Exactly one option is
  correct and answer must match it exactly.
- Do not add outside facts to notes-mode items. In topic mode there are no
  notes; all items are General knowledge.
- Be concise and specific. No shaming or filler."""

FEEDBACK_SYSTEM = """You are a neutral, encouraging study coach.
Return JSON only, with no markdown fences or commentary.
Compare the student's answer with the concept definition and source snippet.
Return:
- covered: array of short phrases the student got right
- missing: array of short phrases that are important but absent
- incorrect: array of {claim, why} for wrong claims; empty if none
- next_step: one concrete action
- score: number from 0 to 1 (0 = no understanding, 1 = complete and accurate)
Judge only against the definition and snippet. If you use general knowledge,
say "General knowledge" explicitly in the why or next_step. Never shame the
student. Keep the feedback specific and brief."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def extract_concepts_prompt(
    *,
    mode: str,
    chunk_text: str,
    heading: str,
    course: str,
) -> tuple[str, str]:
    if mode == "topic":
        user = (
            f"Topic: {chunk_text.removeprefix('Topic:').strip()}\n"
            f"Course/context: {course or 'General'}\n\n"
            'Return a JSON object shaped like {"concepts": [...]}.'
        )
        return TOPIC_SYSTEM, user
    user = (
        f"Course/context: {course or 'General'}\n"
        f"Section: {heading or 'Notes'}\n\n"
        "Notes:\n---\n"
        f"{chunk_text}\n"
        "---\n\n"
        'Return a JSON object shaped like {"concepts": [...]}.'
    )
    return EXTRACT_SYSTEM, user


def make_items_prompt(
    *,
    mode: str,
    concepts: list[dict[str, Any]],
    chunk_text: str,
) -> tuple[str, str]:
    concept_view = [
        {
            "id": c.get("id"),
            "name": c.get("name"),
            "definition": c.get("definition"),
            "difficulty": c.get("difficulty"),
        }
        for c in concepts
    ]
    if mode == "topic":
        user = (
            "Topic:\n---\n"
            f"{chunk_text}\n"
            "---\n\n"
            "Concepts:\n"
            f"{_json(concept_view)}\n\n"
            'Return a JSON object shaped like {"cards": [...], "quiz": [...]}.'
            " There are no notes, so all items are General knowledge: omit source_snippet."
        )
        return ITEMS_SYSTEM, user
    user = (
        "Notes:\n---\n"
        f"{chunk_text}\n"
        "---\n\n"
        "Concepts:\n"
        f"{_json(concept_view)}\n\n"
        'Return a JSON object shaped like {"cards": [...], "quiz": [...]}.'
    )
    return ITEMS_SYSTEM, user


def feedback_prompt(*, concept: dict[str, Any], student_answer: str) -> tuple[str, str]:
    user = (
        "Concept:\n"
        f"{_json({k: concept.get(k) for k in ('name', 'definition', 'source_snippet')})}\n\n"
        "Student answer:\n---\n"
        f"{student_answer}\n"
        "---\n\n"
        'Return a JSON object shaped like '
        '{"covered": [], "missing": [], "incorrect": [], "next_step": "", "score": 0.0}.'
    )
    return FEEDBACK_SYSTEM, user


def repair_concepts_prompt(*, raw: str, mode: str) -> tuple[str, str]:
    system = TOPIC_SYSTEM if mode == "topic" else EXTRACT_SYSTEM
    user = (
        "Your previous response was not valid JSON. Return only a valid JSON "
        'object shaped like {"concepts": [...]}. Do not add commentary.\n\n'
        f"Previous response:\n{raw[:3000]}"
    )
    return system, user


def repair_items_prompt(
    *,
    raw: str,
    mode: str,
    concepts: list[dict[str, Any]],
    chunk_text: str,
) -> tuple[str, str]:
    user = (
        "Your previous item response failed validation. Return only a valid JSON "
        'object shaped like {"cards": [...], "quiz": [...]}. For every item, either '
        "copy a verbatim source_snippet from the notes or omit the snippet and set "
        "grounded:false (General knowledge). Do not add commentary.\n\n"
        f"Concepts:\n{_json(concepts)}\n\n"
        f"Notes:\n---\n{chunk_text[:6000]}\n---\n\n"
        f"Previous response:\n{raw[:2000]}"
    )
    return ITEMS_SYSTEM, user
