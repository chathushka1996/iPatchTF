from uuid import UUID

from sqlalchemy import delete, desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.collection import Collection, CollectionGame
from app.repositories.base import BaseRepository


class CollectionRepository(BaseRepository[Collection]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Collection, session)

    async def get_by_user(
        self, user_id: UUID, skip: int = 0, limit: int = 24
    ) -> list[Collection]:
        result = await self.session.execute(
            select(Collection)
            .options(selectinload(Collection.games))
            .where(Collection.user_id == user_id)
            .order_by(desc(Collection.updated_at))
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_public(self, skip: int = 0, limit: int = 24) -> list[Collection]:
        result = await self.session.execute(
            select(Collection)
            .options(selectinload(Collection.games))
            .where(Collection.is_public.is_(True))
            .order_by(desc(Collection.updated_at))
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def add_game(
        self, collection_id: UUID, game_id: UUID, note: str | None = None
    ) -> CollectionGame:
        max_order_result = await self.session.execute(
            select(CollectionGame.sort_order)
            .where(CollectionGame.collection_id == collection_id)
            .order_by(desc(CollectionGame.sort_order))
            .limit(1)
        )
        max_order = max_order_result.scalar_one_or_none() or 0

        entry = CollectionGame(
            collection_id=collection_id,
            game_id=game_id,
            note=note,
            sort_order=max_order + 1,
        )
        self.session.add(entry)
        await self.session.execute(
            update(Collection)
            .where(Collection.id == collection_id)
            .values(game_count=Collection.game_count + 1)
        )
        await self.session.flush()
        return entry

    async def remove_game(self, collection_id: UUID, game_id: UUID) -> bool:
        result = await self.session.execute(
            delete(CollectionGame).where(
                CollectionGame.collection_id == collection_id,
                CollectionGame.game_id == game_id,
            )
        )
        if result.rowcount > 0:
            await self.session.execute(
                update(Collection)
                .where(Collection.id == collection_id)
                .values(game_count=Collection.game_count - 1)
            )
            await self.session.flush()
            return True
        return False

    async def reorder_games(self, collection_id: UUID, game_ids: list[UUID]) -> None:
        for index, game_id in enumerate(game_ids):
            await self.session.execute(
                update(CollectionGame)
                .where(
                    CollectionGame.collection_id == collection_id,
                    CollectionGame.game_id == game_id,
                )
                .values(sort_order=index)
            )
        await self.session.flush()
