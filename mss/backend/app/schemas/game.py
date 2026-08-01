from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.user import UserBriefResponse

DevelopmentStatus = Literal[
    "concept", "demo", "alpha", "beta", "complete", "discontinued"
]
ContentRating = Literal["G", "PG", "R", "X", "XXX"]
PCGender = Literal["male", "female", "selectable", "genderless", "hermaphrodite"]
DownloadPlatform = Literal["windows", "mac", "linux", "browser", "android"]
TagCategory = Literal[
    "genre",
    "adult_theme",
    "transformation",
    "multimedia",
    "content_warning",
    "platform",
]
GameSortOption = Literal[
    "newest", "updated", "rating", "likes", "title", "trending"
]


class EngineResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    game_count: int = 0


class TagResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    category: str


class DownloadCreate(BaseModel):
    url: str
    label: str = Field(max_length=100)
    file_size_bytes: int | None = Field(default=None, ge=0)
    platform: DownloadPlatform


class DownloadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    url: str
    label: str
    file_size_bytes: int | None = None
    platform: str


class ScreenshotResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    image_url: str
    thumbnail_url: str | None = None
    caption: str | None = None
    sort_order: int = 0


class GameVersionCreate(BaseModel):
    version_string: str = Field(max_length=50)
    changelog: str | None = None
    release_date: date | None = None
    downloads: list[DownloadCreate] = Field(default_factory=list)


class GameVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    version_string: str
    changelog: str | None = None
    release_date: date | None = None
    is_latest: bool = False
    downloads: list[DownloadResponse] = Field(default_factory=list)
    created_at: datetime


# Alias used in GameResponse.latest_version
VersionResponse = GameVersionResponse


class GameCreate(BaseModel):
    title: str = Field(max_length=255)
    engine_id: int
    development_status: DevelopmentStatus
    rating: ContentRating
    original_pc_gender: PCGender
    language: str = Field(default="English", max_length=50)
    is_free: bool = True
    has_purchasable_content: bool = False
    support_url: str | None = Field(default=None, max_length=500)
    synopsis: str | None = None
    plot: str | None = None
    characters: str | None = None
    walkthrough: str | None = None
    tag_ids: list[int] = Field(default_factory=list)
    play_online_url: str | None = Field(default=None, max_length=500)


class GameUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    engine_id: int | None = None
    development_status: DevelopmentStatus | None = None
    rating: ContentRating | None = None
    original_pc_gender: PCGender | None = None
    language: str | None = Field(default=None, max_length=50)
    is_free: bool | None = None
    has_purchasable_content: bool | None = None
    support_url: str | None = Field(default=None, max_length=500)
    synopsis: str | None = None
    plot: str | None = None
    characters: str | None = None
    walkthrough: str | None = None
    tag_ids: list[int] | None = None
    play_online_url: str | None = Field(default=None, max_length=500)


class GameListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    slug: str
    engine_name: str
    author_name: str
    development_status: str
    rating: str
    like_count: int = 0
    review_count: int = 0
    average_score: Decimal = Field(default=Decimal("0.00"))
    thumbnail_url: str | None = None
    created_at: datetime
    updated_at: datetime


class GameResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    slug: str
    synopsis: str | None = None
    plot: str | None = None
    characters: str | None = None
    walkthrough: str | None = None
    engine_id: int
    original_pc_gender: str
    rating: str
    development_status: str
    is_free: bool
    has_purchasable_content: bool
    support_url: str | None = None
    language: str
    play_online_url: str | None = None
    like_count: int = 0
    review_count: int = 0
    average_score: Decimal = Field(default=Decimal("0.00"))
    view_count: int = 0
    created_at: datetime
    updated_at: datetime
    engine: EngineResponse
    author: UserBriefResponse
    tags: list[TagResponse] = Field(default_factory=list)
    screenshots: list[ScreenshotResponse] = Field(default_factory=list)
    latest_version: VersionResponse | None = None


class GameSearchParams(BaseModel):
    q: str | None = None
    engine: list[str] | None = None
    status: list[DevelopmentStatus] | None = None
    genre: list[str] | None = None
    adult_theme: list[str] | None = None
    transformation: list[str] | None = None
    multimedia: list[str] | None = None
    content_warning: list[str] | None = None
    rating: list[ContentRating] | None = None
    pc_gender: list[PCGender] | None = None
    author: str | None = None
    has_play_online: bool | None = None
    min_likes: int | None = Field(default=None, ge=0)
    sort: GameSortOption = "newest"
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=24, ge=1, le=100)
