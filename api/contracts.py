"""Strict, bounded request contracts for the v2 backend boundary."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from functools import lru_cache
import os
from typing import Annotated, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from feedback.v2_settings import V2FeedbackSettings


ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
MediumText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
]
ListText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
DescriptionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, max_length=10_000),
]
ReadmeText = Annotated[str, StringConstraints(max_length=1_000_000)]
ParagraphText = Annotated[str, StringConstraints(max_length=10_000)]
NonNegativeCount = Annotated[int, Field(ge=0, le=9_223_372_036_854_775_807)]
DeltaCount = Annotated[int, Field(ge=-2_147_483_648, le=2_147_483_647)]
UnitScore = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be an ISO-8601 UTC timestamp")
    return value


@lru_cache(maxsize=16)
def _feedback_dwell_bounds(
    minimum_seconds: str | None,
    full_credit_seconds: str | None,
) -> tuple[int, int]:
    # The raw values are cache keys so tests and local processes that update
    # configuration before constructing a contract cannot retain stale bounds.
    settings = V2FeedbackSettings.from_env()
    return settings.dwell_min_ms, settings.dwell_full_credit_ms


def feedback_dwell_bounds() -> tuple[int, int]:
    return _feedback_dwell_bounds(
        os.getenv("FEEDBACK_DWELL_MIN_SECONDS"),
        os.getenv("FEEDBACK_DWELL_FULL_CREDIT_SECONDS"),
    )


class RecommendationContext(StrictModel):
    cold_start: bool = False
    # Locale is reserved for deterministic filtering/experiments; retrieval
    # currently reports it as reserved rather than silently changing ranking.
    locale: str | None = Field(
        default=None,
        min_length=2,
        max_length=32,
        pattern=r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*$",
    )


class RecommendationRequest(StrictModel):
    schema_version: Literal[2]
    generation_id: uuid.UUID
    user_id: uuid.UUID
    feed_version: int = Field(ge=1)
    limit: int = Field(ge=1, le=100)
    exclude_repo_ids: list[uuid.UUID] = Field(default_factory=list, max_length=500)
    context: RecommendationContext

    @field_validator("exclude_repo_ids")
    @classmethod
    def unique_exclusions(cls, value: list[uuid.UUID]) -> list[uuid.UUID]:
        if len(set(value)) != len(value):
            raise ValueError("exclude_repo_ids must be unique")
        return value


EventType = Literal[
    "impression",
    "dwell",
    "readme_open",
    "github_open",
    "like",
    "unlike",
    "dislike",
    "undislike",
    "save",
    "unsave",
    "share",
]


class FeedbackEvent(StrictModel):
    event_id: uuid.UUID
    user_id: uuid.UUID
    repo_id: uuid.UUID
    feedback_version: int = Field(ge=1)
    event_type: EventType
    dwell_ms: int | None = None
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def utc_occurred_at(cls, value: datetime) -> datetime:
        return _utc(value, field_name="occurred_at")

    @model_validator(mode="after")
    def validate_dwell(self):
        if self.event_type == "impression":
            raise ValueError("impressions are offline-only and must not be sent to ML")
        if self.event_type == "dwell":
            minimum, maximum = feedback_dwell_bounds()
            if self.dwell_ms is None or not minimum <= self.dwell_ms <= maximum:
                raise ValueError(
                    f"dwell_ms must be between {minimum} and {maximum}"
                )
        elif self.dwell_ms is not None:
            raise ValueError("only dwell events may carry dwell_ms")
        return self


class FeedbackBatch(StrictModel):
    schema_version: Literal[2]
    events: list[FeedbackEvent] = Field(min_length=1, max_length=100)

    @field_validator("events")
    @classmethod
    def unique_events(cls, value: list[FeedbackEvent]) -> list[FeedbackEvent]:
        if len({event.event_id for event in value}) != len(value):
            raise ValueError("event_id values must be unique within a batch")
        return value


class RepositorySource(StrictModel):
    github_id: str | None = Field(default=None, pattern=r"^[0-9]{1,32}$")
    full_name: str = Field(min_length=3, max_length=256, pattern=r"^[^/\s]+/[^/\s]+$")
    html_url: str | None = Field(
        default=None,
        max_length=2_048,
        validation_alias=AliasChoices("html_url", "url"),
    )
    description: DescriptionText = ""
    primary_language: ShortText = "Unknown"
    languages: list[ListText] = Field(default_factory=list, max_length=100)
    topics: list[ListText] = Field(default_factory=list, max_length=200)
    readme: ReadmeText | None = None
    extracted_paragraphs: list[ParagraphText] = Field(
        default_factory=list, max_length=512
    )
    readme_length: NonNegativeCount = 0
    star_count: NonNegativeCount = 0
    fork_count: NonNegativeCount = 0
    open_issues_count: NonNegativeCount = 0
    pushed_days_ago: Annotated[int, Field(ge=0, le=100_000)] = 999
    delta_3d: DeltaCount = 0
    delta_7d: DeltaCount = 0
    delta_30d: DeltaCount = 0
    mentionable_users_count: NonNegativeCount = 0
    readme_to_codebase_ratio: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    recent_commits: list[datetime] = Field(default_factory=list, max_length=100)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    pushed_at: datetime | None = None
    discovery_category: MediumText | None = None
    discovery_band: MediumText | None = None
    content_hash: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("created_at", "updated_at", "pushed_at")
    @classmethod
    def utc_timestamps(cls, value: datetime | None, info):
        return None if value is None else _utc(value, field_name=info.field_name)

    @field_validator("recent_commits")
    @classmethod
    def utc_commits(cls, value: list[datetime]) -> list[datetime]:
        return [_utc(item, field_name="recent_commits") for item in value]

    @model_validator(mode="after")
    def bounded_readme(self):
        paragraph_size = sum(len(item) for item in self.extracted_paragraphs)
        if paragraph_size > 1_000_000:
            raise ValueError("extracted_paragraphs exceed the 1 MB content limit")
        if self.readme is not None and self.extracted_paragraphs:
            raise ValueError("send readme or extracted_paragraphs, not both")
        return self


class RepositoryJob(StrictModel):
    schema_version: Literal[2]
    job_id: uuid.UUID
    repo_id: uuid.UUID
    content_version: int = Field(ge=1)
    repository: RepositorySource


class RepositoryFeaturePatch(StrictModel):
    star_count: NonNegativeCount | None = None
    fork_count: NonNegativeCount | None = None
    open_issues_count: NonNegativeCount | None = None
    pushed_days_ago: Annotated[int, Field(ge=0, le=100_000)] | None = None
    delta_3d: DeltaCount | None = None
    delta_7d: DeltaCount | None = None
    delta_30d: DeltaCount | None = None
    mentionable_users_count: NonNegativeCount | None = None
    doc_quality: UnitScore | None = None
    code_health: UnitScore | None = None
    activity_score: UnitScore | None = None
    trend_velocity: UnitScore | None = None
    updated_at: datetime | None = None
    pushed_at: datetime | None = None
    discovery_category: MediumText | None = None
    discovery_band: MediumText | None = None

    @field_validator("updated_at", "pushed_at")
    @classmethod
    def utc_timestamps(cls, value: datetime | None, info):
        return None if value is None else _utc(value, field_name=info.field_name)

    @model_validator(mode="after")
    def not_empty(self):
        if not self.model_fields_set:
            raise ValueError("features must contain at least one refreshable field")
        return self


class RepositoryRefreshJob(StrictModel):
    schema_version: Literal[2]
    job_id: uuid.UUID
    repo_id: uuid.UUID
    feature_version: int = Field(ge=1)
    features: RepositoryFeaturePatch


class OnboardingProfile(StrictModel):
    github_username: ShortText | None = None
    username: ShortText | None = None
    full_name: MediumText | None = None
    bio: str | None = Field(default=None, max_length=2_000)
    interests: list[ListText] = Field(default_factory=list, max_length=100)
    topics: list[ListText] = Field(default_factory=list, max_length=100)
    skills: list[ListText] = Field(default_factory=list, max_length=100)
    tech_stack: list[ListText] = Field(default_factory=list, max_length=100)
    avatar_url: str | None = Field(default=None, max_length=2_048)

    @model_validator(mode="after")
    def has_embedding_context(self):
        if not (
            self.bio
            or self.interests
            or self.topics
            or self.skills
            or self.tech_stack
        ):
            raise ValueError(
                "profile must include bio, interests, topics, skills, or tech_stack"
            )
        return self


class OnboardingJob(StrictModel):
    schema_version: Literal[2]
    job_id: uuid.UUID
    user_id: uuid.UUID
    profile_version: int = Field(ge=1)
    profile: OnboardingProfile
