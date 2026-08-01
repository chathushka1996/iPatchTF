from celery import Celery
from celery.schedules import crontab

from app.config import settings

celery_app = Celery(
    "gamevault",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "cleanup-expired-tokens": {
            "task": "app.tasks.cleanup_tasks.cleanup_expired_tokens",
            "schedule": crontab(hour=3, minute=0),
        },
        "cleanup-old-notifications": {
            "task": "app.tasks.cleanup_tasks.cleanup_old_notifications",
            "schedule": crontab(hour=4, minute=0),
        },
        "recalculate-trending-scores": {
            "task": "app.tasks.cleanup_tasks.recalculate_trending_scores",
            "schedule": crontab(minute=0),
        },
        "recalculate-game-scores": {
            "task": "app.tasks.cleanup_tasks.recalculate_game_scores",
            "schedule": crontab(minute="*/30"),
        },
        "generate-recommendations": {
            "task": "app.tasks.cleanup_tasks.generate_recommendations",
            "schedule": crontab(hour=2, minute=0),
        },
    },
)

celery_app.autodiscover_tasks(
    [
        "app.tasks.email_tasks",
        "app.tasks.search_tasks",
        "app.tasks.notification_tasks",
        "app.tasks.cleanup_tasks",
    ]
)
