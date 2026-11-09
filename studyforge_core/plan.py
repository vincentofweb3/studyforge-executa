"""Deterministic study-plan allocation.

Concepts are ordered by prerequisites, then difficulty and current mastery.
New concepts are spread across the learn days within the daily minutes budget;
the final two days are always reserved for review and a mixed quiz. No LLM is
used here, so re-planning is instant when the date or minutes change.
"""

from __future__ import annotations

import heapq
import math
import re
from datetime import date, timedelta
from typing import Any, Iterable

from .models import Plan, PlanDay, utc_now_iso
from .srs import as_date

NEW_CONCEPT_BASE_MINUTES = 8
NEW_CONCEPT_MINUTES_PER_DIFFICULTY = 4
REVIEW_CONCEPT_MINUTES = 4
DEFAULT_PLAN_DAYS = 14


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _concept_ids(concepts: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    by_id = {str(c.get("id")): c for c in concepts if c.get("id")}
    by_name = {_name_key(c.get("name", "")): str(c.get("id")) for c in concepts if c.get("id")}
    return by_id, by_name


def topological_order(concepts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    concepts = [dict(c) for c in concepts]
    by_id, by_name = _concept_ids(concepts)
    indegree: dict[str, int] = {cid: 0 for cid in by_id}
    children: dict[str, set[str]] = {cid: set() for cid in by_id}
    prereqs: dict[str, set[str]] = {cid: set() for cid in by_id}

    for cid, concept in by_id.items():
        for raw in concept.get("prerequisites") or []:
            dep = str(raw)
            if dep not in by_id:
                dep = by_name.get(_name_key(dep), "")
            if dep and dep in by_id and dep != cid:
                prereqs[cid].add(dep)

    for cid, deps in prereqs.items():
        indegree[cid] = len(deps)
        for dep in deps:
            children[dep].add(cid)

    def priority(cid: str) -> tuple[int, int, str]:
        concept = by_id[cid]
        return (
            int(concept.get("difficulty") or 3),
            int(concept.get("mastery_level") or 0),
            str(concept.get("name") or "").lower(),
        )

    heap = [(priority(cid), cid) for cid, degree in indegree.items() if degree == 0]
    heapq.heapify(heap)
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    while heap:
        _prio, cid = heapq.heappop(heap)
        if cid in seen:
            continue
        seen.add(cid)
        ordered.append(by_id[cid])
        for child in sorted(children[cid]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(heap, (priority(child), child))

    if len(ordered) < len(by_id):
        # Break cycles deterministically rather than dropping concepts.
        remaining = [by_id[cid] for cid in by_id if cid not in seen]
        remaining.sort(key=lambda c: (int(c.get("difficulty") or 3), str(c.get("name") or "").lower()))
        ordered.extend(remaining)
    return ordered


def _new_cost(concept: dict[str, Any]) -> int:
    return NEW_CONCEPT_BASE_MINUTES + NEW_CONCEPT_MINUTES_PER_DIFFICULTY * int(
        concept.get("difficulty") or 3
    )


def _date_range(start: date, count: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(count)]


def generate_plan(
    pack: dict[str, Any],
    concepts: Iterable[dict[str, Any]],
    today: str | date | None = None,
) -> Plan:
    start = as_date(today)
    concepts = [dict(c) for c in concepts]
    daily_minutes = int(pack.get("daily_minutes") or 30)
    target = as_date(pack.get("target_date")) if pack.get("target_date") else None
    if target is None:
        target = start + timedelta(days=DEFAULT_PLAN_DAYS - 1)
    elif target < start:
        target = start + timedelta(days=1)
    total_days = max(1, (target - start).days + 1)
    review_days = min(2, total_days)
    learn_days = total_days - review_days

    ordered = topological_order(concepts)
    by_id = {str(c.get("id")): c for c in ordered}
    day_of: dict[str, int] = {}
    days: list[PlanDay] = []

    if learn_days > 0 and ordered:
        per_day = max(1, math.ceil(len(ordered) / learn_days))
        assignments: list[list[dict[str, Any]]] = [[] for _ in range(learn_days)]
        used = [0] * learn_days
        for concept in ordered:
            cost = _new_cost(concept)
            min_day = 0
            for raw in concept.get("prerequisites") or []:
                dep = str(raw)
                if dep not in day_of:
                    key = _name_key(dep)
                    dep = next(
                        (cid for cid, c in by_id.items() if _name_key(c.get("name", "")) == key),
                        "",
                    )
                if dep in day_of:
                    min_day = max(min_day, day_of[dep])
            candidate = None
            for index in range(min_day, learn_days):
                if len(assignments[index]) >= per_day:
                    continue
                if used[index] + cost <= daily_minutes:
                    candidate = index
                    break
            if candidate is None:
                for index in range(min_day, learn_days):
                    if len(assignments[index]) < per_day:
                        candidate = index
                        break
            if candidate is None:
                candidate = min(range(min_day, learn_days), key=lambda i: used[i])
            assignments[candidate].append(concept)
            used[candidate] += cost
            day_of[str(concept.get("id"))] = candidate

        learn_dates = _date_range(start, learn_days)
        for index, assigned in enumerate(assignments):
            if assigned:
                kind = "learn"
                ids = [str(c.get("id")) for c in assigned]
                new_cards = 2 * len(ids)
                review_cards = 0
            else:
                available = [
                    str(c.get("id"))
                    for c in ordered
                    if day_of.get(str(c.get("id")), learn_days) <= index
                ]
                if not available:
                    available = [str(c.get("id")) for c in ordered]
                subset = available[:3]
                kind = "review"
                ids = subset
                new_cards = 0
                review_cards = 2 * len(subset)
            days.append(
                PlanDay(
                    date=learn_dates[index].isoformat(),
                    minutes=daily_minutes,
                    concept_ids=ids,
                    new_cards=new_cards,
                    review_cards=review_cards,
                    kind=kind,
                )
            )
    elif learn_days > 0:
        for index in range(learn_days):
            days.append(
                PlanDay(
                    date=(start + timedelta(days=index)).isoformat(),
                    minutes=daily_minutes,
                    concept_ids=[],
                    new_cards=0,
                    review_cards=0,
                    kind="review",
                )
            )

    review_start_index = len(days)
    all_ids = [str(c.get("id")) for c in ordered]
    if review_days > 0:
        review_dates = _date_range(start + timedelta(days=review_start_index), review_days)
        for index in range(review_days):
            chunk = all_ids[index::review_days] if review_days else all_ids
            kind = "quiz" if index == review_days - 1 else "review"
            days.append(
                PlanDay(
                    date=review_dates[index].isoformat(),
                    minutes=daily_minutes,
                    concept_ids=chunk,
                    new_cards=0,
                    review_cards=2 * len(chunk),
                    kind=kind,
                )
            )

    while len(days) < total_days:
        days.append(
            PlanDay(
                date=(start + timedelta(days=len(days))).isoformat(),
                minutes=daily_minutes,
                concept_ids=[],
                new_cards=0,
                review_cards=0,
                kind="review",
            )
        )

    if ordered:
        verb = "is" if review_days == 1 else "are"
        summary = (
            f"{total_days}-day plan for {len(ordered)} concepts at {daily_minutes} minutes a day. "
            f"The last {review_days} day{'s' if review_days != 1 else ''} {verb} review and quiz."
        )
    else:
        summary = "No concepts to schedule yet. Add notes or a topic to build a plan."
    return Plan(
        pack_id=str(pack.get("id") or ""),
        days=days[:total_days],
        generated_at=utc_now_iso(),
        summary=summary,
    )
