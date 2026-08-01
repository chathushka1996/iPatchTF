from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.user import UserBriefResponse


class ForumCategoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    description: str | None = None
    sort_order: int = 0
    is_locked: bool = False
    thread_count: int = 0
    post_count: int = 0
    last_post_at: datetime | None = None
    subcategories: list[ForumCategoryResponse] | None = None


class ThreadCreate(BaseModel):
    title: str = Field(max_length=300)
    body: str = Field(min_length=1)


class ThreadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    forum_category_id: int
    game_id: UUID | None = None
    user: UserBriefResponse
    title: str
    slug: str
    is_pinned: bool = False
    is_locked: bool = False
    view_count: int = 0
    post_count: int = 0
    last_post_at: datetime | None = None
    created_at: datetime


class PostResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    thread_id: UUID
    user: UserBriefResponse
    parent_id: UUID | None = None
    body: str
    body_html: str | None = None
    is_edited: bool = False
    edited_at: datetime | None = None
    created_at: datetime
    replies: list[PostResponse] | None = None


class ThreadDetailResponse(ThreadResponse):
    posts: list[PostResponse] = Field(default_factory=list)


class PostCreate(BaseModel):
    body: str = Field(min_length=1)
    parent_id: UUID | None = None


class PostUpdate(BaseModel):
    body: str = Field(min_length=1)


class ChatMessageCreate(BaseModel):
    body: str = Field(min_length=1, max_length=500)


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    channel: str
    user: UserBriefResponse
    body: str
    created_at: datetime


ForumCategoryResponse.model_rebuild()
PostResponse.model_rebuild()
