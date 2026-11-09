"""Tool implementations for the StudyForge Executa dispatcher."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from .chunker import chunk_notes, chunk_topic, normalize_notes
from .errors import ToolError
from .llm import (
    assemble_pack,
    extract_text,
    fallback_feedback,
    merge_concepts,
    normalize_concepts,
    normalize_feedback,
    normalize_items,
    parse_json,
)
from .mastery import update_mastery
from .models import (
    DAILY_MINUTES,
    MAX_NOTES_WORDS,
    MIN_NOTES_WORDS,
    Card,
    CardState,
    Chunk,
    Concept,
    DuckResult,
    Pack,
    QuizAttempt,
    QuizItem,
    normalize_date,
    pack_id_for,
    utc_now_iso,
    word_count,
)
from .plan import generate_plan
from .prompts import (
    extract_concepts_prompt,
    feedback_prompt,
    make_items_prompt,
    repair_concepts_prompt,
    repair_items_prompt,
)
from .srs import apply_reviews, build_due_queue, initial_state, review_state
from .verify import verify_snippet

ACTIONS = (
    "prepare_pack",
    "extract_concepts",
    "make_items",
    "merge_concepts",
    "assemble_pack",
    "plan_pack",
    "schedule_review",
    "due_queue",
    "grade_answer",
    "rubber_duck",
    "mastery",
    "ping",
)


def _dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ToolError("invalid_args", f"{name} must be an object.")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ToolError("invalid_args", f"{name} must be an array.")
    return value


def _pack(value: Any) -> dict[str, Any]:
    return Pack.from_dict(_dict(value, "pack")).to_dict()


def _chunk(value: Any) -> dict[str, Any]:
    return Chunk.from_dict(_dict(value, "chunk")).to_dict()


def _concept(value: Any) -> dict[str, Any]:
    return Concept.from_dict(_dict(value, "concept")).to_dict()


def _card(value: Any) -> dict[str, Any]:
    return Card.from_dict(_dict(value, "card")).to_dict()


def _quiz(value: Any) -> dict[str, Any]:
    return QuizItem.from_dict(_dict(value, "quiz item")).to_dict()


def _state(value: Any) -> dict[str, Any]:
    return CardState.from_dict(_dict(value, "state")).to_dict()


def _empty_usage() -> dict[str, int]:
    return {"calls": 0, "inputTokens": 0, "outputTokens": 0, "totalTokens": 0}


def _usage_from(result: dict[str, Any]) -> dict[str, int]:
    usage = result.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = int(usage.get("inputTokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("outputTokens") or usage.get("output_tokens") or 0)
    total = int(usage.get("totalTokens") or usage.get("total_tokens") or input_tokens + output_tokens)
    return {
        "calls": 1,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": total,
    }


def _add_usage(total: dict[str, int], extra: dict[str, int]) -> dict[str, int]:
    for key in total:
        total[key] += int(extra.get(key) or 0)
    return total


def _sample(
    sampler: Any,
    invoke_id: str | None,
    *,
    system: str,
    user: str,
    max_tokens: int,
    timeout: float = 60.0,
) -> tuple[str, dict[str, int]]:
    if sampler is None:
        raise ToolError("sampling_unavailable", "The study tool needs host LLM access. Try again.")
    result = sampler.create_message(
        messages=[{"role": "user", "content": {"type": "text", "text": user}}],
        max_tokens=max_tokens,
        system_prompt=system,
        temperature=0.2,
        response_format={"type": "json_object"},
        on_unsupported="text",
        timeout=timeout,
        invoke_id=invoke_id,
    )
    return extract_text(result), _usage_from(result)


def _prepare_pack(payload: dict[str, Any], _sampler: Any = None, _invoke_id: Any = None) -> dict[str, Any]:
    notes = str(payload.get("notes") or "")
    topic = str(payload.get("topic") or "")
    mode = str(payload.get("mode") or ("notes" if notes.strip() else "topic")).lower()
    if mode not in ("notes", "topic"):
        raise ToolError("invalid_mode", "Choose notes or topic mode.")

    if mode == "notes":
        normalized = normalize_notes(notes)
        words = word_count(normalized)
        if words < MIN_NOTES_WORDS:
            raise ToolError(
                "notes_too_short",
                f"Add at least {MIN_NOTES_WORDS} words of notes, or switch to topic mode.",
                {"word_count": words},
            )
        if words > MAX_NOTES_WORDS:
            raise ToolError(
                "notes_too_long",
                f"Notes are capped at {MAX_NOTES_WORDS:,} words. Split them into two packs.",
                {"word_count": words},
            )
        source = normalized
    else:
        normalized = topic.strip()
        if not normalized:
            raise ToolError("topic_required", "Add a topic to build a study pack.")
        if len(normalized) > 500:
            raise ToolError("topic_too_long", "Keep the topic under 500 characters.")
        source = normalized

    try:
        daily_minutes = int(payload.get("daily_minutes") or 30)
    except (TypeError, ValueError) as exc:
        raise ToolError("invalid_minutes", "Choose 15, 30, 60 or 90 minutes a day.") from exc
    if daily_minutes not in DAILY_MINUTES:
        raise ToolError("invalid_minutes", "Choose 15, 30, 60 or 90 minutes a day.")

    title = str(payload.get("title") or "").strip()[:120]
    course = str(payload.get("course") or "").strip()[:80]
    target_date = normalize_date(payload.get("target_date"))
    target_type = str(payload.get("target_type") or "none").lower()
    if not title:
        title = f"Topic: {normalized}" if mode == "topic" else "Study pack"
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    pack = Pack(
        id=pack_id_for(mode, source, "", title, course),
        title=title,
        course=course or "General",
        target_date=target_date,
        daily_minutes=daily_minutes,
        mode=mode,
        target_type=target_type if target_type in ("exam", "interview", "certification", "none") else "none",
        created_at=utc_now_iso(),
        source_hash=source_hash,
    ).validate()

    chunks = chunk_notes(source, pack.id) if mode == "notes" else chunk_topic(source, pack.id)
    if not chunks:
        raise ToolError("empty_notes", "Those notes produced no readable sections. Add more detail.")
    return {
        "pack": pack.to_dict(),
        "chunks": [chunk.to_dict() for chunk in chunks],
        "word_count": word_count(source) if mode == "notes" else word_count(source),
        "source_hash": source_hash,
    }


def _extract_concepts(payload: dict[str, Any], sampler: Any, invoke_id: str | None) -> dict[str, Any]:
    pack = _pack(payload.get("pack"))
    chunk = _chunk(payload.get("chunk"))
    mode = str(pack.get("mode") or "notes")
    system, user = extract_concepts_prompt(
        mode=mode,
        chunk_text=str(chunk.get("text") or ""),
        heading=str(chunk.get("heading") or ""),
        course=str(pack.get("course") or ""),
    )
    max_tokens = 5_000 if mode == "topic" else 4_096
    usage = _empty_usage()
    raw_text = ""
    try:
        raw_text, extra = _sample(
            sampler, invoke_id, system=system, user=user, max_tokens=max_tokens
        )
        usage = _add_usage(usage, extra)
        data = parse_json(raw_text)
    except ToolError:
        system, user = repair_concepts_prompt(raw=raw_text, mode=mode)
        raw_text, extra = _sample(
            sampler, invoke_id, system=system, user=user, max_tokens=max_tokens
        )
        usage = _add_usage(usage, extra)
        data = parse_json(raw_text)

    concepts = normalize_concepts(
        data,
        pack_id=str(pack.get("id") or ""),
        chunk_id=str(chunk.get("id") or ""),
        mode=mode,
        chunk_text=str(chunk.get("text") or ""),
    )
    if not concepts:
        raise ToolError(
            "no_concepts",
            "The model did not find any concepts in that section. Add more detail or try topic mode.",
        )
    return {"concepts": concepts, "usage": usage}


def _make_items(payload: dict[str, Any], sampler: Any, invoke_id: str | None) -> dict[str, Any]:
    pack = _pack(payload.get("pack"))
    concepts = [_concept(c) for c in _list(payload.get("concepts"), "concepts")]
    if not concepts:
        return {"cards": [], "quiz": [], "usage": _empty_usage()}
    chunk_text = str(payload.get("chunk_text") or "")
    mode = str(pack.get("mode") or "notes")
    system, user = make_items_prompt(mode=mode, concepts=concepts, chunk_text=chunk_text)
    usage = _empty_usage()
    raw_text = ""
    try:
        raw_text, extra = _sample(
            sampler, invoke_id, system=system, user=user, max_tokens=6_000
        )
        usage = _add_usage(usage, extra)
        data = parse_json(raw_text)
    except ToolError:
        system, user = repair_items_prompt(
            raw=raw_text, mode=mode, concepts=concepts, chunk_text=chunk_text
        )
        raw_text, extra = _sample(
            sampler, invoke_id, system=system, user=user, max_tokens=6_000
        )
        usage = _add_usage(usage, extra)
        data = parse_json(raw_text)

    items = normalize_items(data, pack=pack, concepts=concepts, chunk_text=chunk_text)
    if not items["cards"] and not items["quiz"]:
        raise ToolError(
            "no_items",
            "The model did not return usable cards for that section. Retry the stage.",
        )
    return {**items, "usage": usage}


def _merge_concepts_action(payload: dict[str, Any], _sampler: Any = None, _invoke_id: Any = None) -> dict[str, Any]:
    pack = _pack(payload.get("pack"))
    concepts = [_concept(c) for c in _list(payload.get("concepts"), "concepts")]
    for concept in concepts:
        concept["pack_id"] = str(pack.get("id") or concept.get("pack_id") or "")
    return merge_concepts(concepts)


def _assemble_pack_action(payload: dict[str, Any], _sampler: Any = None, _invoke_id: Any = None) -> dict[str, Any]:
    pack = _pack(payload.get("pack"))
    chunks = [_chunk(c) for c in _list(payload.get("chunks"), "chunks")]
    concepts = [_concept(c) for c in _list(payload.get("concepts"), "concepts")]
    cards = [_card(c) for c in _list(payload.get("cards"), "cards")]
    quiz = [_quiz(q) for q in _list(payload.get("quiz"), "quiz")]
    return assemble_pack(pack=pack, chunks=chunks, concepts=concepts, cards=cards, quiz=quiz)


def _plan_pack_action(payload: dict[str, Any], _sampler: Any = None, _invoke_id: Any = None) -> dict[str, Any]:
    pack = _pack(payload.get("pack"))
    concepts = [_concept(c) for c in _list(payload.get("concepts"), "concepts")]
    plan = generate_plan(pack, concepts, payload.get("today"))
    return {"plan": plan.to_dict()}


def _schedule_review_action(payload: dict[str, Any], _sampler: Any = None, _invoke_id: Any = None) -> dict[str, Any]:
    today = payload.get("today")
    if "reviews" in payload:
        states = [_state(s) for s in _list(payload.get("states"), "states")]
        reviews = _list(payload.get("reviews"), "reviews")
        return {"states": [s.to_dict() for s in apply_reviews(states, reviews, today)]}
    if "state" in payload:
        state = _state(payload.get("state"))
        rating = str(payload.get("rating") or "")
        return {"state": review_state(state, rating, today).to_dict()}
    card_id = str(payload.get("card_id") or "")
    if not card_id:
        raise ToolError("invalid_args", "Provide a state, or a card_id and rating.")
    state = initial_state(card_id, today)
    return {"state": review_state(state, str(payload.get("rating") or ""), today).to_dict()}


def _due_queue_action(payload: dict[str, Any], _sampler: Any = None, _invoke_id: Any = None) -> dict[str, Any]:
    states = [_state(s) for s in _list(payload.get("states"), "states")]
    cards = [_card(c) for c in _list(payload.get("cards"), "cards")]
    concepts = [_concept(c) for c in _list(payload.get("concepts"), "concepts")]
    try:
        daily_minutes = int(payload.get("daily_minutes") or 30)
    except (TypeError, ValueError) as exc:
        raise ToolError("invalid_minutes", "Daily minutes must be a number.") from exc
    return build_due_queue(states, cards, concepts, daily_minutes, payload.get("today"))


def _grade_answer(payload: dict[str, Any], sampler: Any, invoke_id: str | None) -> dict[str, Any]:
    concept = _concept(payload.get("concept"))
    student_answer = str(payload.get("student_answer") or "")[:4000]
    system, user = feedback_prompt(concept=concept, student_answer=student_answer)
    usage = _empty_usage()
    try:
        raw_text, extra = _sample(
            sampler, invoke_id, system=system, user=user, max_tokens=1_200
        )
        usage = _add_usage(usage, extra)
        data = parse_json(raw_text)
        feedback = normalize_feedback(data, concept, student_answer)
        return {"feedback": feedback.to_dict(), "usage": usage}
    except Exception:
        feedback = fallback_feedback(concept, student_answer)
        return {"feedback": feedback.to_dict(), "usage": usage, "fallback": True}


def _rubber_duck(payload: dict[str, Any], sampler: Any, invoke_id: str | None) -> dict[str, Any]:
    concept = _concept(payload.get("concept"))
    student_answer = str(payload.get("student_explanation") or "")[:4000]
    system, user = feedback_prompt(concept=concept, student_answer=student_answer)
    usage = _empty_usage()
    try:
        raw_text, extra = _sample(
            sampler, invoke_id, system=system, user=user, max_tokens=1_200
        )
        usage = _add_usage(usage, extra)
        data = parse_json(raw_text)
        feedback = normalize_feedback(data, concept, student_answer)
        return {"feedback": feedback.to_dict(), "usage": usage}
    except Exception:
        feedback = fallback_feedback(concept, student_answer)
        return {"feedback": feedback.to_dict(), "usage": usage, "fallback": True}


def _mastery_action(payload: dict[str, Any], _sampler: Any = None, _invoke_id: Any = None) -> dict[str, Any]:
    concepts = [_concept(c) for c in _list(payload.get("concepts"), "concepts")]
    cards = [_card(c) for c in _list(payload.get("cards"), "cards")]
    states = [_state(s) for s in _list(payload.get("states"), "states")]
    attempts = [QuizAttempt(**a).to_dict() for a in _list(payload.get("attempts"), "attempts")]
    ducks = [DuckResult(**d).to_dict() for d in _list(payload.get("duck_results"), "duck_results")]
    return {"concepts": update_mastery(concepts, cards, states, attempts, ducks)}


def _ping(payload: dict[str, Any] | None = None, _sampler: Any = None, _invoke_id: Any = None) -> dict[str, Any]:
    return {"pong": True, "tool": "studyforge", "version": "0.1.0"}


_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "prepare_pack": _prepare_pack,
    "extract_concepts": _extract_concepts,
    "make_items": _make_items,
    "merge_concepts": _merge_concepts_action,
    "assemble_pack": _assemble_pack_action,
    "plan_pack": _plan_pack_action,
    "schedule_review": _schedule_review_action,
    "due_queue": _due_queue_action,
    "grade_answer": _grade_answer,
    "rubber_duck": _rubber_duck,
    "mastery": _mastery_action,
    "ping": _ping,
}


def dispatch(
    tool: str,
    arguments: dict[str, Any] | None,
    sampler: Any = None,
    invoke_id: str | None = None,
) -> dict[str, Any]:
    if tool == "ping":
        return _ping()
    if tool != "studyforge":
        raise ToolError("unknown_tool", f"Unknown tool: {tool}")
    args = arguments or {}
    action = str(args.get("action") or "")
    payload = args.get("payload") or {}
    if action not in _HANDLERS:
        raise ToolError("unknown_action", f"Unknown StudyForge action: {action}")
    if not isinstance(payload, dict):
        raise ToolError("invalid_args", "payload must be an object.")
    return _HANDLERS[action](payload, sampler, invoke_id)
