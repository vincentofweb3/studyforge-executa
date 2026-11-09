"""Mastery score and level.

Formula (documented in tests and shown in the UI legend):

    score = 0.5 * average(card interval factor)
          + 0.3 * recent quiz accuracy
          + 0.2 * latest rubber-duck score

A missing component contributes 0 (no renormalisation). Card interval factor
is min(1, interval_days / 21). Quiz accuracy uses the latest 10 attempts for
the concept. Levels: 0 New (0), 1 Shaky (<0.35), 2 Learning (<0.6),
3 Solid (<0.85), 4 Locked in (>=0.85).
"""

from __future__ import annotations

from typing import Any, Iterable

from .models import CardState, MASTERY_LABELS

CARD_WEIGHT = 0.5
QUIZ_WEIGHT = 0.3
DUCK_WEIGHT = 0.2
FULL_INTERVAL_DAYS = 21.0
RECENT_QUIZ_LIMIT = 10


def interval_factor(interval_days: int | float) -> float:
    return min(1.0, max(0.0, float(interval_days) / FULL_INTERVAL_DAYS))


def _as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "to_dict"):
        return item.to_dict()
    return {}


def _state_map(states: Iterable[Any]) -> dict[str, CardState]:
    out: dict[str, CardState] = {}
    for state in states:
        item = state if isinstance(state, CardState) else CardState.from_dict(_as_dict(state))
        out[item.card_id] = item
    return out


def quiz_accuracy(attempts: Iterable[Any], concept_id: str, limit: int = RECENT_QUIZ_LIMIT) -> float:
    relevant = [
        a
        for a in (_as_dict(x) for x in attempts)
        if str(a.get("concept_id") or "") == concept_id
    ]
    relevant.sort(key=lambda a: str(a.get("at") or ""), reverse=True)
    recent = relevant[:limit]
    if not recent:
        return 0.0
    return sum(1 for a in recent if bool(a.get("correct"))) / len(recent)


def rubber_duck_score(results: Iterable[Any], concept_id: str) -> float:
    relevant = [
        r
        for r in (_as_dict(x) for x in results)
        if str(r.get("concept_id") or "") == concept_id
    ]
    relevant.sort(key=lambda r: str(r.get("at") or ""), reverse=True)
    if not relevant:
        return 0.0
    try:
        return max(0.0, min(1.0, float(relevant[0].get("score") or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def level_for(score: float) -> int:
    if score <= 0:
        return 0
    if score < 0.35:
        return 1
    if score < 0.6:
        return 2
    if score < 0.85:
        return 3
    return 4


def concept_mastery(
    concept: dict[str, Any],
    cards: Iterable[dict[str, Any]],
    states: Iterable[Any],
    attempts: Iterable[Any] | None = None,
    duck_results: Iterable[Any] | None = None,
) -> dict[str, Any]:
    concept_id = str(concept.get("id") or "")
    state_map = _state_map(states)
    concept_cards = [c for c in cards if str(c.get("concept_id") or "") == concept_id]
    factors: list[float] = []
    for card in concept_cards:
        card_id = str(card.get("id") or "")
        state = state_map.get(card_id)
        factors.append(interval_factor(state.interval_days if state else 0))
    card_factor = sum(factors) / len(factors) if factors else 0.0
    accuracy = quiz_accuracy(attempts or [], concept_id)
    duck = rubber_duck_score(duck_results or [], concept_id)
    score = CARD_WEIGHT * card_factor + QUIZ_WEIGHT * accuracy + DUCK_WEIGHT * duck
    score = round(max(0.0, min(1.0, score)), 4)
    level = level_for(score)
    return {
        "score": score,
        "level": level,
        "label": MASTERY_LABELS[level],
        "components": {
            "card_factor": round(card_factor, 4),
            "quiz_accuracy": round(accuracy, 4),
            "duck_score": round(duck, 4),
        },
    }


def update_mastery(
    concepts: Iterable[dict[str, Any]],
    cards: Iterable[dict[str, Any]],
    states: Iterable[Any],
    attempts: Iterable[Any] | None = None,
    duck_results: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    cards_list = list(cards)
    states_list = list(states)
    attempts_list = list(attempts or [])
    duck_list = list(duck_results or [])
    out: list[dict[str, Any]] = []
    for concept in concepts:
        updated = dict(concept)
        result = concept_mastery(updated, cards_list, states_list, attempts_list, duck_list)
        updated["mastery_level"] = result["level"]
        updated["mastery_score"] = result["score"]
        updated["mastery_label"] = result["label"]
        out.append(updated)
    return out
