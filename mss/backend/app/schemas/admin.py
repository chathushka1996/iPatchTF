from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

UserRoleLiteral = Literal["user", "moderator", "admin"]
ReportStatusLiteral = Literal["pending", "reviewed", "resolved", "dismissed"]
TagCategoryLiteral = Literal[
    "genre",
    "adult_theme",
    "transformation",
    "multimedia",
    "content_warning",
    "platform",
]


class AdminDashboardResponse(BaseModel):
    new_users_today: int = Field(ge=0)
    new_users_week: int = Field(ge=0)
    new_users_month: int = Field(ge=0)
    new_games: int = Field(ge=0)
    new_reviews: int = Field(ge=0)
    pending_reports: int = Field(ge=0)
    active_users: int = Field(ge=0)


class RoleChangeRequest(BaseModel):
    role: UserRoleLiteral


class ReportUpdateRequest(BaseModel):
    status: ReportStatusLiteral
    resolution_note: str | None = None


class TagCreate(BaseModel):
    name: str = Field(max_length=100)
    slug: str | None = Field(default=None, max_length=120)
    category: TagCategoryLiteral
    description: str | None = None


class EngineCreate(BaseModel):
    name: str = Field(max_length=100)
    slug: str | None = Field(default=None, max_length=120)
