from typing import Any

from app.tasks import celery_app


@celery_app.task(name="app.tasks.email_tasks.send_email_task")
def send_email_task(
    to: str,
    subject: str,
    template: str,
    context: dict[str, Any],
) -> None:
    import asyncio

    from app.services.email_service import EmailService

    async def _send() -> None:
        service = EmailService()
        await service.send_email(to, subject, template, context)

    asyncio.run(_send())
