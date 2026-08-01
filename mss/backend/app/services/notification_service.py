import json
from typing import Any
from uuid import UUID

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.notification import Notification
from app.repositories.notification_repo import NotificationRepository


class NotificationService:
    CHANNEL_PREFIX = "notifications:user:"

    def __init__(
        self,
        session: AsyncSession,
        redis: aioredis.Redis | None = None,
    ) -> None:
        self.session = session
        self.repo = NotificationRepository(session)
        self._redis = redis

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        return self._redis

    async def create_notification(
        self,
        user_id: UUID,
        type: str,
        title: str,
        body: str,
        link: str | None = None,
    ) -> Notification:
        notification = await self.repo.create(
            {
                "user_id": user_id,
                "type": type,
                "title": title,
                "body": body,
                "link": link,
            }
        )

        redis = await self._get_redis()
        message = {
            "id": str(notification.id),
            "type": type,
            "title": title,
            "body": body,
            "link": link,
            "created_at": notification.created_at.isoformat(),
        }
        await redis.publish(
            f"{self.CHANNEL_PREFIX}{user_id}",
            json.dumps(message),
        )
        return notification

    async def get_notifications(
        self, user_id: UUID, page: int = 1, per_page: int = 24
    ) -> tuple[list[Notification], int]:
        skip = (page - 1) * per_page
        items = await self.repo.get_by_user(user_id, skip=skip, limit=per_page)
        total = await self.repo.count({"user_id": user_id})
        return items, total

    async def get_unread_count(self, user_id: UUID) -> int:
        return await self.repo.get_unread_count(user_id)

    async def mark_read(self, ids: list[UUID], user_id: UUID) -> int:
        return await self.repo.mark_read(ids, user_id)

    async def mark_all_read(self, user_id: UUID) -> int:
        return await self.repo.mark_all_read(user_id)
