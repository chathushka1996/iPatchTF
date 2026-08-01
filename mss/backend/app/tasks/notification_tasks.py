from typing import Any
from uuid import UUID

from app.tasks import celery_app


@celery_app.task(name="app.tasks.notification_tasks.send_notification_task")
def send_notification_task(
    user_id: str,
    type: str,
    title: str,
    body: str,
    link: str | None = None,
) -> None:
    import asyncio

    from app.database import async_session_factory
    from app.services.notification_service import NotificationService

    async def _send() -> None:
        async with async_session_factory() as session:
            service = NotificationService(session)
            await service.create_notification(
                UUID(user_id), type, title, body, link
            )
            await session.commit()

    asyncio.run(_send())


@celery_app.task(name="app.tasks.notification_tasks.send_batch_notifications")
def send_batch_notifications(
    user_ids: list[str],
    type: str,
    title: str,
    body: str,
    link: str | None = None,
) -> None:
    for user_id in user_ids:
        send_notification_task.delay(user_id, type, title, body, link)
