from uuid import UUID

from app.tasks import celery_app


@celery_app.task(name="app.tasks.search_tasks.sync_game_to_search")
def sync_game_to_search(game_id: str) -> None:
    import asyncio

    from app.database import async_session_factory
    from app.services.search_service import SearchService

    async def _sync() -> None:
        async with async_session_factory() as session:
            service = SearchService(session)
            await service.sync_game(UUID(game_id))
            await session.commit()

    asyncio.run(_sync())


@celery_app.task(name="app.tasks.search_tasks.remove_game_from_search")
def remove_game_from_search(game_id: str) -> None:
    import asyncio

    from app.services.search_service import SearchService

    async def _remove() -> None:
        service = SearchService()
        await service.delete_game(UUID(game_id))

    asyncio.run(_remove())


@celery_app.task(name="app.tasks.search_tasks.full_reindex")
def full_reindex() -> None:
    import asyncio

    from sqlalchemy import select

    from app.database import async_session_factory
    from app.models.game import Game
    from app.services.search_service import SearchService

    async def _reindex() -> None:
        async with async_session_factory() as session:
            service = SearchService(session)
            await service.configure_index()
            result = await session.execute(select(Game).where(Game.is_approved.is_(True)))
            games = result.scalars().all()
            for game in games:
                await service.sync_game(game.id)
            await session.commit()

    asyncio.run(_reindex())
