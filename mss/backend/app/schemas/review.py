from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.user import UserBriefResponse


class ReviewCreate(BaseModel):
    score: int = Field(ge=1, le=10)
    body: str = Field(min_length=1)


class ReviewUpdate(BaseModel):
    score: int | None = Field(default=None, ge=1, le=10)
    body: str | None = Field(default=None, min_length=1)


class ReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    game_id: UUID
    user: UserBriefResponse
    version_reviewed: str | None = None
    score: int
    body: str
    helpful_count: int = 0
    not_helpful_count: int = 0
    is_edited: bool = False
    created_at: datetime
    updated_at: datetime


class ReviewVoteRequest(BaseModel):
    is_helpful: bool
