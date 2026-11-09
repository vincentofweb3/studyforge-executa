"""SM-2-lite scheduler.

Ratings: Again=0, Hard=3, Good=4, Easy=5 (stored as labels).

Again: reps=0, lapses+1, interval=1, ease=max(1.3, ease-0.2)
Hard:  interval=max(1, round(interval*1.2)), ease=max(1.3, ease-0.15)
Good:  interval=1 (first), 3 (second), else round(interval*ease)
Easy:  interval=3 (first), 6 (second), else round(interval*ease*1.3), ease+0.15

Rounding is half-up in both Python and TypeScript so the shared vectors agree.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Iterable

from .errors import ToolError
from .models import CardState

RATINGS = ("again", "hard", "good", "easy")
REVIEW_SECONDS = 8
NEW_SECONDS = 20
MIN_EASE = 1.3


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _round_ease(value: float) -> float:
    return math.floor(max(MIN_EASE, value) * 10_000 + 0.5) / 10_000


def as_date(value: str | date | None = None) -> date:
    if isinstance(value, date):
        return value
    if not value:
        return date.today()
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ToolError("invalid_date", "Use a date in YYYY-MM-DD format.") from exc


def initial_state(card_id: str, today: str | date | None = None) -> CardState:
    day = as_date(today)
    return CardState(card_id=card_id, ease=2.5, interval_days=0, due=day.isoformat(), reps=0, lapses=0)


def review_state(state: CardState | dict[str, Any], rating: str, today: str | date | None = None) -> CardState:
    current = state if isinstance(state, CardState) else CardState.from_dict(state)
    rating = str(rating).lower()
    if rating not in RATINGS:
        raise ToolError("invalid_rating", "Rating must be again, hard, good or easy.")

    ease = max(MIN_EASE, float(current.ease))
    interval = max(0, int(current.interval_days))
    reps = max(0, int(current.reps))
    lapses = max(0, int(current.lapses))

    if rating == "again":
        reps = 0
        lapses += 1
        interval = 1
        ease = _round_ease(ease - 0.2)
    elif rating == "hard":
        interval = max(1, _round_half_up(interval * 1.2))
        ease = _round_ease(ease - 0.15)
        reps += 1
    elif rating == "good":
        if reps == 0:
            interval = 1
        elif reps == 1:
            interval = 3
        else:
            interval = _round_half_up(interval * ease)
        reps += 1
    else:  # easy
        if reps == 0:
            interval = 3
        elif reps == 1:
            interval = 6
        else:
            interval = _round_half_up(interval * ease * 1.3)
        ease = _round_ease(ease + 0.15)
        reps += 1

    interval = max(1, interval)
    due = as_date(today) + timedelta(days=interval)
    return CardState(
        card_id=current.card_id,
        ease=ease,
        interval_days=interval,
        due=due.isoformat(),
        reps=reps,
        lapses=lapses,
        last_rating=rating,
    )


def apply_reviews(
    states: Iterable[CardState | dict[str, Any]],
    reviews: Iterable[dict[str, Any]],
    today: str | date | None = None,
) -> list[CardState]:
    by_id: dict[str, CardState] = {}
    for state in states:
        item = state if isinstance(state, CardState) else CardState.from_dict(state)
        by_id[item.card_id] = item
    for review in reviews:
        card_id = str(review.get("card_id") or "")
        if not card_id:
            raise ToolError("invalid_review", "Each review needs a card_id.")
        current = by_id.get(card_id) or initial_state(card_id, today)
        by_id[card_id] = review_state(current, str(review.get("rating") or ""), today)
    return list(by_id.values())


def _concept_map(concepts: Iterable[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for concept in concepts or []:
        if isinstance(concept, dict) and concept.get("id"):
            out[str(concept["id"])] = concept
    return out


def build_due_queue(
    states: Iterable[CardState | dict[str, Any]],
    cards: Iterable[dict[str, Any]],
    concepts: Iterable[dict[str, Any]] | None,
    daily_minutes: int,
    today: str | date | None = None,
) -> dict[str, Any]:
    day = as_date(today)
    state_map: dict[str, CardState] = {}
    for state in states:
        item = state if isinstance(state, CardState) else CardState.from_dict(state)
        state_map[item.card_id] = item
    concept_map = _concept_map(concepts)

    reviews: list[tuple[float, str, str]] = []
    new: list[tuple[int, str, str]] = []
    for card in cards:
        card_id = str(card.get("id") or "")
        if not card_id:
            continue
        state = state_map.get(card_id) or initial_state(card_id, day)
        if state.reps == 0:
            concept = concept_map.get(str(card.get("concept_id") or ""), {})
            difficulty = int(concept.get("difficulty") or 3)
            new.append((difficulty, card_id, state.due))
        elif state.due <= day.isoformat():
            reviews.append((state.ease, state.due, card_id))

    reviews.sort()
    new.sort()

    budget = int(daily_minutes) * 60
    used = 0
    picked_reviews: list[str] = []
    picked_new: list[str] = []
    for _ease, _due, card_id in reviews:
        if picked_reviews and used + REVIEW_SECONDS > budget:
            break
        picked_reviews.append(card_id)
        used += REVIEW_SECONDS
    for _difficulty, card_id, _due in new:
        if used + NEW_SECONDS > budget:
            break
        picked_new.append(card_id)
        used += NEW_SECONDS

    return {
        "reviews": picked_reviews,
        "new": picked_new,
        "total": len(picked_reviews) + len(picked_new),
        "estimated_minutes": round(used / 60, 1),
        "budget_seconds": budget,
    }
