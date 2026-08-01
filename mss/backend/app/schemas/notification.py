from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class NotificationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    type: str
    title: str
    body: str | None = None
    link: str | None = None
    is_read: bool = False
    created_at: datetime


class MarkReadRequest(BaseModel):
    notification_ids: list[UUID] | None = Field(
        default=None,
        description="If empty or omitted, mark all notifications as read.",
    )


class UnreadCountResponse(BaseModel):
    count: int = Field(ge=0)
