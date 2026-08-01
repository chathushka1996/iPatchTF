from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

UploadPurpose = Literal["avatar", "screenshot", "game_file", "forum_attachment"]


class PresignRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=1, max_length=100)
    purpose: UploadPurpose


class PresignResponse(BaseModel):
    upload_url: str
    object_key: str
    expires_in: int = Field(ge=1)
