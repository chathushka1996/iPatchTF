from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import delete, desc, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.game import Game, GameFollow, GameLike, GameTag, Tag
from app.repositories.base import BaseRepository


class GameRepository(BaseRepository[Game]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Game, session)

    async def get_by_slug(self, slug: str) -> Game | None:
        result = await self.session.execute(
            select(Game)
            .options(
                selectinload(Game.engine),
                selectinload(Game.tags),
                selectinload(Game.versions),
                selectinload(Game.screenshots),
            )
            .where(Game.slug == slug)
        )
        return result.scalar_one_or_none()

    async def search(
        self,
        filters: dict[str, Any],
        pagination: dict[str, int],
    ) -> tuple[list[Game], int]:
        query = select(Game).where(Game.is_approved.is_(True))
        count_query = select(func.count()).select_from(Game).where(Game.is_approved.is_(True))

        if q := filters.get("q"):
            pattern = f"%{q}%"
            query = query.where(
                or_(Game.title.ilike(pattern), Game.synopsis.ilike(pattern))
            )
            count_query = count_query.where(
                or_(Game.title.ilike(pattern), Game.synopsis.ilike(pattern))
            )

        if engine_id := filters.get("engine_id"):
            query = query.where(Game.engine_id == engine_id)
            count_query = count_query.where(Game.engine_id == engine_id)

        if author_id := filters.get("author_id"):
            query = query.where(Game.author_id == author_id)
            count_query = count_query.where(Game.author_id == author_id)

        if status := filters.get("status"):
            query = query.where(Game.development_status == status)
            count_query = count_query.where(Game.development_status == status)

        if rating := filters.get("rating"):
            query = query.where(Game.rating == rating)
            count_query = count_query.where(Game.rating == rating)

        if filters.get("is_featured"):
            query = query.where(Game.is_featured.is_(True))
            count_query = count_query.where(Game.is_featured.is_(True))

        sort = filters.get("sort", "newest")
        sort_map = {
            "newest": desc(Game.created_at),
            "updated": desc(Game.updated_at),
            "rating": desc(Game.average_score),
            "likes": desc(Game.like_count),
            "title": Game.title.asc(),
            "trending": desc(Game.view_count + Game.like_count * 3),
        }
        order_by = sort_map.get(sort, desc(Game.created_at))
        query = query.order_by(order_by)

        skip = pagination.get("skip", 0)
        limit = pagination.get("limit", 24)

        total_result = await self.session.execute(count_query)
        total = total_result.scalar_one()

        result = await self.session.execute(query.offset(skip).limit(limit))
        return list(result.scalars().all()), total

    async def get_featured(self, limit: int = 12) -> list[Game]:
        result = await self.session.execute(
            select(Game)
            .where(Game.is_featured.is_(True), Game.is_approved.is_(True))
            .order_by(desc(Game.updated_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_trending(self, limit: int = 12) -> list[Game]:
        week_ago = datetime.now(UTC) - timedelta(days=7)
        result = await self.session.execute(
            select(Game)
            .where(Game.is_approved.is_(True), Game.updated_at >= week_ago)
            .order_by(desc(Game.like_count * 3 + Game.view_count))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_recent(self, limit: int = 12) -> list[Game]:
        result = await self.session.execute(
            select(Game)
            .where(Game.is_approved.is_(True))
            .order_by(desc(Game.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_recently_updated(self, limit: int = 12) -> list[Game]:
        result = await self.session.execute(
            select(Game)
            .where(Game.is_approved.is_(True))
            .order_by(desc(Game.updated_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_author(
        self, user_id: UUID, skip: int = 0, limit: int = 24
    ) -> list[Game]:
        result = await self.session.execute(
            select(Game)
            .where(Game.author_id == user_id)
            .order_by(desc(Game.created_at))
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def toggle_like(self, game_id: UUID, user_id: UUID) -> bool:
        result = await self.session.execute(
            select(GameLike).where(
                GameLike.game_id == game_id,
                GameLike.user_id == user_id,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            await self.session.execute(
                delete(GameLike).where(
                    GameLike.game_id == game_id,
                    GameLike.user_id == user_id,
                )
            )
            await self.session.execute(
                update(Game)
                .where(Game.id == game_id)
                .values(like_count=Game.like_count - 1)
            )
            await self.session.flush()
            return False

        self.session.add(GameLike(game_id=game_id, user_id=user_id))
        await self.session.execute(
            update(Game).where(Game.id == game_id).values(like_count=Game.like_count + 1)
        )
        await self.session.flush()
        return True

    async def follow_game(self, game_id: UUID, user_id: UUID) -> GameFollow:
        follow = GameFollow(game_id=game_id, user_id=user_id)
        self.session.add(follow)
        await self.session.flush()
        return follow

    async def unfollow_game(self, game_id: UUID, user_id: UUID) -> bool:
        result = await self.session.execute(
            delete(GameFollow).where(
                GameFollow.game_id == game_id,
                GameFollow.user_id == user_id,
            )
        )
        await self.session.flush()
        return result.rowcount > 0

    async def is_following(self, game_id: UUID, user_id: UUID) -> bool:
        result = await self.session.execute(
            select(GameFollow).where(
                GameFollow.game_id == game_id,
                GameFollow.user_id == user_id,
            )
        )
        return result.scalar_one_or_none() is not None

    async def get_similar(self, game_id: UUID, limit: int = 6) -> list[Game]:
        tag_ids_result = await self.session.execute(
            select(GameTag.tag_id).where(GameTag.game_id == game_id)
        )
        tag_ids = list(tag_ids_result.scalars().all())
        if not tag_ids:
            result = await self.session.execute(
                select(Game)
                .where(Game.id != game_id, Game.is_approved.is_(True))
                .order_by(desc(Game.like_count))
                .limit(limit)
            )
            return list(result.scalars().all())

        result = await self.session.execute(
            select(Game)
            .join(GameTag, GameTag.game_id == Game.id)
            .where(GameTag.tag_id.in_(tag_ids), Game.id != game_id, Game.is_approved.is_(True))
            .group_by(Game.id)
            .order_by(desc(func.count(GameTag.tag_id)))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def increment_view_count(self, game_id: UUID) -> None:
        await self.session.execute(
            update(Game)
            .where(Game.id == game_id)
            .values(view_count=Game.view_count + 1)
        )
        await self.session.flush()
