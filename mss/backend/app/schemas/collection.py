from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.game import GameListResponse
from app.schemas.user import UserBriefResponse


class CollectionCreate(BaseModel):
    name: str = Field(max_length=200)
    description: str | None = None
    is_public: bool = True


class CollectionUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    description: str | None = None
    is_public: bool | None = None


class CollectionGameAdd(BaseModel):
    game_id: UUID
    note: str | None = None


class CollectionGameResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    game: GameListResponse
    added_at: datetime
    sort_order: int = 0
    note: str | None = None


class CollectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user: UserBriefResponse
    name: str
    description: str | None = None
    is_public: bool
    game_count: int = 0
    created_at: datetime
    updated_at: datetime


class CollectionDetailResponse(CollectionResponse):
    games: list[CollectionGameResponse] = Field(default_factory=list)


class ReorderRequest(BaseModel):
    game_ids: list[UUID] = Field(min_length=1)
