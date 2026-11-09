"""Typed data model for StudyForge.

The tool boundary is JSON, but every object that reaches storage or the UI is
built through one of these dataclasses so counts, ranges and labels stay
consistent. LLM output is normalised in :mod:`llm` before construction.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from .errors import ToolError

MIN_NOTES_WORDS = 80
MAX_NOTES_WORDS = 12_000
MAX_CARDS = 120
MAX_QUIZ = 40
DAILY_MINUTES = (15, 30, 60, 90)
TARGET_TYPES = ("exam", "interview", "certification", "none")
CARD_KINDS = (
    "definition",
    "compare",
    "cloze",
    "complexity",
    "predict_output",
    "spot_bug",
)
TRACKS = (
    "databases",
    "networking",
    "algorithms",
    "operating_systems",
    "system_design",
    "cloud_devops",
    "security",
    "languages",
    "other",
)
MASTERY_LABELS = ("New", "Shaky", "Learning", "Solid", "Locked in")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def stable_id(prefix: str, *parts: str, length: int = 10) -> str:
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def pack_id_for(mode: str, notes: str, topic: str, title: str, course: str) -> str:
    return stable_id("pack", mode, notes, topic, title, course, length=12)


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError) as exc:
        raise ToolError("invalid_date", "Use a target date in YYYY-MM-DD format.") from exc


def normalize_date(value: str | None) -> str | None:
    parsed = parse_date(value)
    return parsed.isoformat() if parsed else None


@dataclass
class Pack:
    id: str
    title: str
    course: str
    target_date: str | None
    daily_minutes: int
    mode: str
    target_type: str
    created_at: str
    source_hash: str
    concept_count: int = 0
    card_count: int = 0
    quiz_count: int = 0

    def validate(self) -> "Pack":
        if self.mode not in ("notes", "topic"):
            raise ToolError("invalid_mode", "Choose notes or topic mode.")
        if self.daily_minutes not in DAILY_MINUTES:
            raise ToolError("invalid_minutes", "Choose 15, 30, 60 or 90 minutes a day.")
        if self.target_type not in TARGET_TYPES:
            raise ToolError("invalid_target_type", "Choose a valid study target.")
        if self.target_date is not None:
            self.target_date = normalize_date(self.target_date)
        if not self.title.strip():
            self.title = "Untitled study pack"
        if not self.course.strip():
            self.course = "General"
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Pack":
        return cls(**{k: raw.get(k) for k in cls.__dataclass_fields__}).validate()


@dataclass
class Chunk:
    id: str
    pack_id: str
    index: int
    text: str
    heading: str
    text_hash: str
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Chunk":
        return cls(**{k: raw.get(k) for k in cls.__dataclass_fields__})


@dataclass
class Concept:
    id: str
    pack_id: str
    name: str
    definition: str
    difficulty: int
    prerequisites: list[str] = field(default_factory=list)
    chunk_id: str = ""
    track: str = "other"
    source_snippet: str | None = None
    source_verified: bool = False
    mastery_level: int = 0
    mastery_score: float = 0.0
    mastery_label: str = "New"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Concept":
        data = {k: raw.get(k) for k in cls.__dataclass_fields__}
        data["prerequisites"] = list(data.get("prerequisites") or [])
        return cls(**data)


@dataclass
class Card:
    id: str
    concept_id: str
    kind: str
    front: str
    back: str
    code: str | None = None
    language: str | None = None
    chunk_id: str = ""
    source_snippet: str | None = None
    grounded: bool = False
    label: str = "General knowledge"
    not_executed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Card":
        return cls(**{k: raw.get(k) for k in cls.__dataclass_fields__})


@dataclass
class QuizItem:
    id: str
    concept_id: str
    kind: str
    prompt: str
    options: list[str] = field(default_factory=list)
    answer: str = ""
    explanation: str = ""
    rubric: list[str] = field(default_factory=list)
    chunk_id: str = ""
    source_snippet: str | None = None
    grounded: bool = False
    label: str = "General knowledge"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QuizItem":
        data = {k: raw.get(k) for k in cls.__dataclass_fields__}
        data["options"] = list(data.get("options") or [])
        data["rubric"] = list(data.get("rubric") or [])
        return cls(**data)


@dataclass
class CardState:
    card_id: str
    ease: float = 2.5
    interval_days: int = 0
    due: str = ""
    reps: int = 0
    lapses: int = 0
    last_rating: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CardState":
        data = {k: raw.get(k) for k in cls.__dataclass_fields__}
        data["ease"] = float(data.get("ease") or 2.5)
        data["interval_days"] = int(data.get("interval_days") or 0)
        data["reps"] = int(data.get("reps") or 0)
        data["lapses"] = int(data.get("lapses") or 0)
        return cls(**data)


@dataclass
class PlanDay:
    date: str
    minutes: int
    concept_ids: list[str]
    new_cards: int
    review_cards: int
    kind: str = "learn"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlanDay":
        data = {k: raw.get(k) for k in cls.__dataclass_fields__}
        data["concept_ids"] = list(data.get("concept_ids") or [])
        return cls(**data)


@dataclass
class Plan:
    pack_id: str
    days: list[PlanDay]
    generated_at: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Plan":
        return cls(
            pack_id=raw.get("pack_id", ""),
            days=[PlanDay.from_dict(d) for d in raw.get("days") or []],
            generated_at=raw.get("generated_at", ""),
            summary=raw.get("summary", ""),
        )


@dataclass
class Feedback:
    covered: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    incorrect: list[dict[str, str]] = field(default_factory=list)
    next_step: str = ""
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Feedback":
        return cls(
            covered=[str(x) for x in raw.get("covered") or []],
            missing=[str(x) for x in raw.get("missing") or []],
            incorrect=[
                {"claim": str(x.get("claim", "")), "why": str(x.get("why", ""))}
                for x in raw.get("incorrect") or []
                if isinstance(x, dict)
            ],
            next_step=str(raw.get("next_step") or ""),
            score=max(0.0, min(1.0, float(raw.get("score") or 0.0))),
        )


@dataclass
class QuizAttempt:
    concept_id: str
    correct: bool
    at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DuckResult:
    concept_id: str
    score: float
    at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
