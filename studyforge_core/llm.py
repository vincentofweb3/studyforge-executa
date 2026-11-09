"""Parsing and normalising LLM output.

The model is untrusted. Every field is cleaned, range-checked and, for
notes-mode items, checked against the source chunk with :mod:`verify`. The
assembler does a second verification pass so a retry or a merge cannot smuggle
an ungrounded citation into the pack.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Iterable

from .errors import ToolError
from .models import (
    CARD_KINDS,
    MAX_CARDS,
    MAX_QUIZ,
    TRACKS,
    Card,
    Concept,
    Feedback,
    QuizItem,
    stable_id,
)
from .verify import prose_tokens, verify_snippet

STOPWORDS = set(
    """a an the and or but if then than that this these those is are was were be been being
    to of in on for with as by from at it its their your you we they he she i not no can
    could should would may might must do does did have has had will just about into over
    under between during before after above below up down out off again further once here
    there when where why how all any both each few more most other some such only own same
    so too very s t don now""".split()
)


def extract_text(result: dict[str, Any]) -> str:
    content = (result or {}).get("content")
    if isinstance(content, dict):
        return str(content.get("text") or "")
    if isinstance(content, list):
        return "".join(
            str(block.get("text") or "") for block in content if isinstance(block, dict)
        )
    return ""


def parse_json(text: str) -> Any:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    starts = [i for i in (raw.find("{"), raw.find("[")) if i >= 0]
    if not starts:
        raise ToolError("llm_output_invalid", "The model did not return JSON. Try again.")
    start = min(starts)
    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack:
                break
            stack.pop()
            if not stack:
                candidate = raw[start : index + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ToolError("llm_output_invalid", "The model returned malformed JSON. Try again.") from exc
    raise ToolError("llm_output_invalid", "The model returned incomplete JSON. Try again.")


def _clean(value: Any, limit: int = 1000) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", "").strip()[:limit]


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _coerce_feedback(data: Any, concept: dict[str, Any], student_answer: str) -> Feedback:
    if not isinstance(data, dict):
        return fallback_feedback(concept, student_answer)
    incorrect = [
        {"claim": _clean(item.get("claim"), 300), "why": _clean(item.get("why"), 500)}
        for item in _list(data.get("incorrect"))
        if isinstance(item, dict) and (_clean(item.get("claim")) or _clean(item.get("why")))
    ]
    try:
        score = float(data.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return Feedback(
        covered=[_clean(x, 160) for x in _list(data.get("covered")) if _clean(x)],
        missing=[_clean(x, 160) for x in _list(data.get("missing")) if _clean(x)],
        incorrect=incorrect,
        next_step=_clean(data.get("next_step"), 500),
        score=max(0.0, min(1.0, score)),
    )


def _key_terms(definition: str) -> list[str]:
    terms: list[str] = []
    for token in prose_tokens(definition):
        if len(token) > 2 and token not in STOPWORDS and token not in terms:
            terms.append(token)
    return terms


def fallback_feedback(concept: dict[str, Any], student_answer: str) -> Feedback:
    definition = _clean(concept.get("definition"), 1000)
    terms = _key_terms(definition)
    answer_tokens = set(prose_tokens(student_answer))
    covered = [term for term in terms if term in answer_tokens][:8]
    missing = [term for term in terms if term not in answer_tokens][:8]
    score = (len(covered) / len(terms)) if terms else 0.0
    if not student_answer.strip():
        covered = []
        missing = terms[:6]
        score = 0.0
    return Feedback(
        covered=covered,
        missing=missing,
        incorrect=[],
        next_step="Review the definition, then explain it again in one or two sentences.",
        score=round(max(0.0, min(1.0, score)), 4),
    )


def normalize_feedback(data: Any, concept: dict[str, Any], student_answer: str) -> Feedback:
    feedback = _coerce_feedback(data, concept, student_answer)
    if not feedback.covered and not feedback.missing and not feedback.incorrect and not feedback.next_step:
        return fallback_feedback(concept, student_answer)
    if not feedback.next_step:
        feedback.next_step = "Review the definition, then explain it again in one or two sentences."
    return feedback


def normalize_concepts(
    data: Any,
    *,
    pack_id: str,
    chunk_id: str,
    mode: str,
    chunk_text: str,
) -> list[dict[str, Any]]:
    raw_list = data.get("concepts") if isinstance(data, dict) else data
    if not isinstance(raw_list, list):
        raise ToolError("llm_output_invalid", "The model did not return a concept list. Try again.")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        name = _clean(raw.get("name"), 100)
        definition = _clean(raw.get("definition"), 600)
        if not name or not definition:
            continue
        key = _name_key(name)
        if key in seen:
            continue
        seen.add(key)
        difficulty = _clamp_int(raw.get("difficulty"), 1, 5, 3)
        track = _clean(raw.get("track"), 40).lower()
        if track not in TRACKS:
            track = "other"
        prerequisites = [
            _clean(item, 100)
            for item in _list(raw.get("prerequisites"))
            if _clean(item)
        ][:8]
        snippet = _clean(raw.get("source_snippet"), 800) or None
        verified = False
        if mode != "notes":
            snippet = None
        elif snippet:
            result = verify_snippet(snippet, chunk_text)
            verified = bool(result["grounded"])
            if not verified:
                snippet = None
        concept = Concept(
            id=stable_id("concept", pack_id, chunk_id, key),
            pack_id=pack_id,
            name=name,
            definition=definition,
            difficulty=difficulty,
            prerequisites=prerequisites,
            chunk_id=chunk_id,
            track=track,
            source_snippet=snippet,
            source_verified=verified,
        )
        out.append(concept.to_dict())
    return out


def _concept_lookup(concepts: Iterable[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for concept in concepts:
        cid = str(concept.get("id") or "")
        if cid:
            by_id[cid] = concept
        by_name[_name_key(concept.get("name", ""))] = concept
    return by_id, by_name


def _find_concept(
    raw: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
    concepts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    cid = _clean(raw.get("concept_id") or raw.get("conceptId"), 80)
    if cid in by_id:
        return by_id[cid]
    name = _clean(raw.get("concept_name") or raw.get("conceptName"), 120)
    if _name_key(name) in by_name:
        return by_name[_name_key(name)]
    if len(concepts) == 1:
        return concepts[0]
    return None


def normalize_items(
    data: Any,
    *,
    pack: dict[str, Any],
    concepts: list[dict[str, Any]],
    chunk_text: str,
) -> dict[str, list[dict[str, Any]]]:
    mode = str(pack.get("mode") or "notes")
    pack_id = str(pack.get("id") or "")
    cards_raw: list[Any] = []
    quiz_raw: list[Any] = []
    if isinstance(data, dict):
        cards_raw = _list(data.get("cards"))
        quiz_raw = _list(data.get("quiz"))
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if "front" in item or item.get("type") == "card":
                cards_raw.append(item)
            elif "prompt" in item or item.get("type") == "quiz":
                quiz_raw.append(item)
    by_id, by_name = _concept_lookup(concepts)

    cards: list[dict[str, Any]] = []
    seen_cards: set[str] = set()
    for raw in cards_raw:
        if not isinstance(raw, dict):
            continue
        concept = _find_concept(raw, by_id, by_name, concepts)
        if not concept:
            continue
        front = _clean(raw.get("front"), 600)
        back = _clean(raw.get("back"), 1200)
        if not front or not back:
            continue
        signature = _name_key(front)
        if signature in seen_cards:
            continue
        seen_cards.add(signature)
        kind = _clean(raw.get("kind"), 40).lower()
        if kind not in CARD_KINDS:
            kind = "definition"
        code = _clean(raw.get("code"), 2000) or None
        language = _clean(raw.get("language"), 40) or None
        snippet = _clean(raw.get("source_snippet"), 800) or None
        code_kind = bool(code) or kind in ("predict_output", "spot_bug")
        grounded = False
        if mode != "notes":
            snippet = None
        elif snippet:
            grounded = bool(verify_snippet(snippet, chunk_text, code=code_kind)["grounded"])
            if not grounded:
                snippet = None
        card = Card(
            id=stable_id("card", pack_id, str(concept.get("id")), signature),
            concept_id=str(concept.get("id")),
            kind=kind,
            front=front,
            back=back,
            code=code,
            language=language,
            chunk_id=str(concept.get("chunk_id") or ""),
            source_snippet=snippet,
            grounded=grounded,
            label="From your notes" if grounded else "General knowledge",
            not_executed=bool(raw.get("not_executed")) or kind in ("predict_output", "spot_bug"),
        )
        cards.append(card.to_dict())

    quiz: list[dict[str, Any]] = []
    seen_quiz: set[str] = set()
    for raw in quiz_raw:
        if not isinstance(raw, dict):
            continue
        concept = _find_concept(raw, by_id, by_name, concepts)
        if not concept:
            continue
        prompt = _clean(raw.get("prompt"), 800)
        if not prompt:
            continue
        signature = _name_key(prompt)
        if signature in seen_quiz:
            continue
        seen_quiz.add(signature)
        kind = _clean(raw.get("kind"), 20).lower()
        if kind not in ("mcq", "short"):
            kind = "mcq"
        options = [_clean(item, 300) for item in _list(raw.get("options")) if _clean(item)]
        answer = _clean(raw.get("answer"), 600)
        explanation = _clean(raw.get("explanation"), 1200)
        rubric = [_clean(item, 200) for item in _list(raw.get("rubric")) if _clean(item)]
        if kind == "mcq":
            if len(options) < 2 or not answer:
                continue
            normalized = {_name_key(option): option for option in options}
            if _name_key(answer) not in normalized:
                if answer not in options:
                    continue
            else:
                answer = normalized[_name_key(answer)]
            rubric = []
        snippet = _clean(raw.get("source_snippet"), 800) or None
        grounded = False
        if mode != "notes":
            snippet = None
        elif snippet:
            grounded = bool(verify_snippet(snippet, chunk_text)["grounded"])
            if not grounded:
                snippet = None
        item = QuizItem(
            id=stable_id("quiz", pack_id, str(concept.get("id")), signature),
            concept_id=str(concept.get("id")),
            kind=kind,
            prompt=prompt,
            options=options,
            answer=answer,
            explanation=explanation,
            rubric=rubric,
            chunk_id=str(concept.get("chunk_id") or ""),
            source_snippet=snippet,
            grounded=grounded,
            label="From your notes" if grounded else "General knowledge",
        )
        quiz.append(item.to_dict())

    return {"cards": cards, "quiz": quiz}


def merge_concepts(concepts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    for raw in concepts:
        name = _clean(raw.get("name"), 100)
        if not name:
            continue
        key = _name_key(name)
        if key not in merged:
            item = dict(raw)
            item["name"] = name
            item["prerequisites"] = [
                _clean(p, 100) for p in _list(item.get("prerequisites")) if _clean(p)
            ]
            merged[key] = item
            continue
        existing = merged[key]
        if raw.get("id"):
            aliases[str(raw["id"])] = str(existing.get("id"))
        existing["difficulty"] = max(int(existing.get("difficulty") or 3), int(raw.get("difficulty") or 3))
        existing["mastery_level"] = max(
            int(existing.get("mastery_level") or 0), int(raw.get("mastery_level") or 0)
        )
        existing["prerequisites"] = list(
            dict.fromkeys(existing.get("prerequisites", []) + _list(raw.get("prerequisites")))
        )
        if raw.get("source_verified") and not existing.get("source_verified"):
            existing["definition"] = _clean(raw.get("definition"), 600) or existing.get("definition")
            existing["source_snippet"] = raw.get("source_snippet")
            existing["source_verified"] = True
            existing["chunk_id"] = raw.get("chunk_id") or existing.get("chunk_id")
        elif len(_clean(raw.get("definition"), 600)) > len(existing.get("definition", "")):
            existing["definition"] = _clean(raw.get("definition"), 600)
        if existing.get("track") in (None, "", "other") and raw.get("track") not in (None, "", "other"):
            existing["track"] = raw.get("track")

    by_id = {str(c.get("id")): c for c in merged.values() if c.get("id")}
    name_to_id = {_name_key(c.get("name", "")): str(c.get("id")) for c in merged.values() if c.get("id")}
    alias_to_final = {alias: final for alias, final in aliases.items() if alias != final}

    for concept in merged.values():
        mapped: list[str] = []
        for raw_dep in concept.get("prerequisites") or []:
            dep = str(raw_dep)
            final = alias_to_final.get(dep)
            if not final:
                if dep in by_id:
                    final = dep
                else:
                    final = name_to_id.get(_name_key(dep), "")
            if final and final != concept.get("id") and final not in mapped:
                mapped.append(final)
        concept["prerequisites"] = mapped

    graph: dict[str, set[str]] = defaultdict(set)

    def reaches(start: str, target: str) -> bool:
        stack = [start]
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(graph.get(node, ()))
        return False

    for concept in merged.values():
        cid = str(concept.get("id"))
        kept: list[str] = []
        for dep in concept.get("prerequisites") or []:
            if not reaches(cid, dep):
                kept.append(dep)
                graph[dep].add(cid)
        concept["prerequisites"] = kept

    return {"concepts": list(merged.values()), "aliases": alias_to_final}


def assemble_pack(
    *,
    pack: dict[str, Any],
    chunks: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    quiz: list[dict[str, Any]],
) -> dict[str, Any]:
    mode = str(pack.get("mode") or "notes")
    chunk_map = {str(c.get("id")): c for c in chunks}
    concept_map = {str(c.get("id")): c for c in concepts}

    def source_for(item: dict[str, Any], concept_id: str) -> str:
        concept = concept_map.get(concept_id) or {}
        chunk_id = str(item.get("chunk_id") or concept.get("chunk_id") or "")
        chunk = chunk_map.get(chunk_id) or {}
        return str(chunk.get("text") or "")

    verified_cards: list[dict[str, Any]] = []
    for raw in cards:
        concept_id = str(raw.get("concept_id") or "")
        concept = concept_map.get(concept_id)
        if not concept:
            continue
        item = dict(raw)
        snippet = item.get("source_snippet")
        code_kind = bool(item.get("code")) or item.get("kind") in ("predict_output", "spot_bug")
        if mode == "notes" and not snippet and item.get("code"):
            snippet = item.get("code")
        grounded = False
        if mode == "notes" and snippet:
            grounded = bool(verify_snippet(snippet, source_for(item, concept_id), code=code_kind)["grounded"])
        item["grounded"] = grounded
        item["label"] = "From your notes" if grounded else "General knowledge"
        item["source_snippet"] = snippet if grounded else None
        verified_cards.append(item)

    verified_quiz: list[dict[str, Any]] = []
    for raw in quiz:
        concept_id = str(raw.get("concept_id") or "")
        concept = concept_map.get(concept_id)
        if not concept:
            continue
        item = dict(raw)
        snippet = item.get("source_snippet")
        grounded = bool(
            mode == "notes" and snippet and verify_snippet(snippet, source_for(item, concept_id))["grounded"]
        )
        item["grounded"] = grounded
        item["label"] = "From your notes" if grounded else "General knowledge"
        item["source_snippet"] = snippet if grounded else None
        verified_quiz.append(item)

    def priority(item: dict[str, Any], order: int) -> tuple[int, int, int]:
        concept = concept_map.get(str(item.get("concept_id") or ""), {})
        return (
            -int(concept.get("difficulty") or 3),
            -len(concept.get("prerequisites") or []),
            order,
        )

    cards_sorted = [item for _p, item in sorted((priority(c, i), c) for i, c in enumerate(verified_cards))]
    quiz_sorted = [item for _p, item in sorted((priority(q, i), q) for i, q in enumerate(verified_quiz))]
    cards_capped = cards_sorted[:MAX_CARDS]
    quiz_capped = quiz_sorted[:MAX_QUIZ]

    kept_ids = {str(c.get("concept_id")) for c in cards_capped}
    concepts_kept = [dict(c) for c in concepts if str(c.get("id")) in kept_ids]
    kept_ids = {str(c.get("id")) for c in concepts_kept}
    for concept in concepts_kept:
        concept["prerequisites"] = [
            p for p in concept.get("prerequisites") or [] if str(p) in kept_ids
        ]

    final_pack = dict(pack)
    final_pack["concept_count"] = len(concepts_kept)
    final_pack["card_count"] = len([c for c in cards_capped if str(c.get("concept_id")) in kept_ids])
    final_pack["quiz_count"] = len([q for q in quiz_capped if str(q.get("concept_id")) in kept_ids])
    return {
        "pack": final_pack,
        "concepts": concepts_kept,
        "cards": [c for c in cards_capped if str(c.get("concept_id")) in kept_ids],
        "quiz": [q for q in quiz_capped if str(q.get("concept_id")) in kept_ids],
    }
