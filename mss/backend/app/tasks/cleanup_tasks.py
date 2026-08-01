from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

from sqlalchemy import delete, select, update

from app.tasks import celery_app


@celery_app.task(name="app.tasks.cleanup_tasks.cleanup_expired_tokens")
def cleanup_expired_tokens() -> None:
    import asyncio

    import redis.asyncio as aioredis

    from app.config import settings

    async def _cleanup() -> None:
        redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        await redis.close()

    asyncio.run(_cleanup())


@celery_app.task(name="app.tasks.cleanup_tasks.cleanup_old_notifications")
def cleanup_old_notifications() -> None:
    import asyncio

    from app.database import async_session_factory
    from app.models.notification import Notification

    async def _cleanup() -> None:
        cutoff = datetime.now(UTC) - timedelta(days=90)
        async with async_session_factory() as session:
            await session.execute(
                delete(Notification).where(
                    Notification.is_read.is_(True),
                    Notification.created_at < cutoff,
                )
            )
            await session.commit()

    asyncio.run(_cleanup())


@celery_app.task(name="app.tasks.cleanup_tasks.recalculate_trending_scores")
def recalculate_trending_scores() -> None:
    import asyncio

    from datetime import timedelta

    from app.database import async_session_factory
    from app.models.game import Game

    async def _recalculate() -> None:
        week_ago = datetime.now(UTC) - timedelta(days=7)
        async with async_session_factory() as session:
            result = await session.execute(
                select(Game).where(
                    Game.is_approved.is_(True),
                    Game.updated_at >= week_ago,
                )
            )
            games = result.scalars().all()
            for game in games:
                trending_score = (
                    game.like_count * 3
                    + game.view_count * 0.1
                    + game.review_count * 10
                )
                await session.execute(
                    update(Game)
                    .where(Game.id == game.id)
                    .values(view_count=game.view_count)
                )
            await session.commit()

    asyncio.run(_recalculate())


@celery_app.task(name="app.tasks.cleanup_tasks.recalculate_game_scores")
def recalculate_game_scores(game_id: str | None = None) -> None:
    import asyncio

    from app.database import async_session_factory
    from app.models.game import Game
    from app.models.review import Review
    from app.repositories.review_repo import ReviewRepository

    async def _recalculate() -> None:
        async with async_session_factory() as session:
            repo = ReviewRepository(session)
            if game_id:
                game_ids = [UUID(game_id)]
            else:
                result = await session.execute(select(Game.id))
                game_ids = list(result.scalars().all())

            for gid in game_ids:
                avg = await repo.average_score_for_game(gid)
                count = await repo.count_by_game(gid)
                await session.execute(
                    update(Game)
                    .where(Game.id == gid)
                    .values(
                        average_score=Decimal(str(round(avg, 2))),
                        review_count=count,
                    )
                )
            await session.commit()

    asyncio.run(_recalculate())


@celery_app.task(name="app.tasks.cleanup_tasks.generate_recommendations")
def generate_recommendations() -> None:
    import asyncio

    import redis.asyncio as aioredis

    from app.config import settings
    from app.database import async_session_factory
    from app.models.game import Game
    from app.repositories.game_repo import GameRepository

    async def _generate() -> None:
        redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        async with async_session_factory() as session:
            repo = GameRepository(session)
            result = await session.execute(
                select(Game).where(Game.is_approved.is_(True)).limit(100)
            )
            games = result.scalars().all()
            for game in games:
                similar = await repo.get_similar(game.id, limit=6)
                key = f"recommendations:{game.id}"
                await redis.delete(key)
                if similar:
                    await redis.rpush(
                        key, *[str(g.id) for g in similar]
                    )
                    await redis.expire(key, 86400)
        await redis.close()

    asyncio.run(_generate())
