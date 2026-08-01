from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, RootModel


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=8)


class UserLogin(BaseModel):
    email: str
    password: str


class UserUpdate(BaseModel):
    display_name: str | None = Field(default=None, max_length=100)
    bio: str | None = None
    website: str | None = Field(default=None, max_length=500)
    location: str | None = Field(default=None, max_length=100)
    social_discord: str | None = Field(default=None, max_length=100)
    social_twitter: str | None = Field(default=None, max_length=100)
    social_github: str | None = Field(default=None, max_length=100)
    patreon_url: str | None = Field(default=None, max_length=500)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    username: str
    email: str
    display_name: str | None = None
    avatar_url: str | None = None
    bio: str | None = None
    website: str | None = None
    location: str | None = None
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime


class UserPublicResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    username: str
    display_name: str | None = None
    avatar_url: str | None = None
    bio: str | None = None
    website: str | None = None
    location: str | None = None
    social_discord: str | None = None
    social_twitter: str | None = None
    social_github: str | None = None
    patreon_url: str | None = None
    created_at: datetime
    game_count: int = 0
    review_count: int = 0
    follower_count: int = 0
    following_count: int = 0


class UserBriefResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    username: str
    display_name: str | None = None
    avatar_url: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(min_length=8)


class NotificationPreferencesUpdate(RootModel[dict[str, Any]]):
    """JSONB-like notification preference payload."""

    root: dict[str, Any]
