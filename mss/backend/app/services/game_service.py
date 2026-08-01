from datetime import date
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import ForbiddenError, NotFoundError
from app.models.game import Game, GameVersion
from app.repositories.forum_repo import ForumRepository
from app.repositories.game_repo import GameRepository
from app.utils.slugify import generate_unique_slug


class GameService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.game_repo = GameRepository(session)
        self.forum_repo = ForumRepository(session)

    async def create_game(self, data: dict[str, Any], user_id: UUID) -> Game:
        slug = await generate_unique_slug(data["title"], Game, self.session)
        game_data = {
            **data,
            "slug": slug,
            "author_id": user_id,
        }
        game = await self.game_repo.create(game_data)

        await self.forum_repo.create_thread(
            {
                "forum_category_id": data.get("forum_category_id", 1),
                "game_id": game.id,
                "user_id": user_id,
                "title": f"Discussion: {game.title}",
                "slug": f"game-{slug}",
            }
        )

        from app.tasks.search_tasks import sync_game_to_search

        sync_game_to_search.delay(str(game.id))
        return game

    async def update_game(
        self, slug: str, data: dict[str, Any], user_id: UUID
    ) -> Game:
        game = await self.game_repo.get_by_slug(slug)
        if not game:
            raise NotFoundError(detail="Game not found")
        if game.author_id != user_id:
            raise ForbiddenError(detail="Not authorized to update this game")

        if "title" in data and data["title"] != game.title:
            data["slug"] = await generate_unique_slug(data["title"], Game, self.session)

        updated = await self.game_repo.update(game.id, data)
        if not updated:
            raise NotFoundError(detail="Game not found")

        from app.tasks.search_tasks import sync_game_to_search

        sync_game_to_search.delay(str(game.id))
        return updated

    async def get_game(self, slug: str) -> Game:
        game = await self.game_repo.get_by_slug(slug)
        if not game:
            raise NotFoundError(detail="Game not found")
        await self.game_repo.increment_view_count(game.id)
        return game

    async def list_games(
        self, filters: dict[str, Any], page: int = 1, per_page: int = 24
    ) -> tuple[list[Game], int]:
        pagination = {"skip": (page - 1) * per_page, "limit": per_page}
        return await self.game_repo.search(filters, pagination)

    async def delete_game(self, slug: str) -> bool:
        game = await self.game_repo.get_by_slug(slug)
        if not game:
            raise NotFoundError(detail="Game not found")

        deleted = await self.game_repo.delete(game.id)
        if deleted:
            from app.tasks.search_tasks import remove_game_from_search

            remove_game_from_search.delay(str(game.id))
        return deleted

    async def add_version(self, slug: str, data: dict[str, Any]) -> GameVersion:
        game = await self.game_repo.get_by_slug(slug)
        if not game:
            raise NotFoundError(detail="Game not found")

        from sqlalchemy import update

        await self.session.execute(
            update(GameVersion)
            .where(GameVersion.game_id == game.id)
            .values(is_latest=False)
        )

        version = GameVersion(
            game_id=game.id,
            version_string=data["version_string"],
            changelog=data.get("changelog"),
            release_date=data.get("release_date", date.today()),
            is_latest=True,
        )
        self.session.add(version)
        await self.session.flush()
        await self.session.refresh(version)
        return version

    async def get_versions(self, slug: str) -> list[GameVersion]:
        game = await self.game_repo.get_by_slug(slug)
        if not game:
            raise NotFoundError(detail="Game not found")
        return game.versions

    async def toggle_like(self, slug: str, user_id: UUID) -> bool:
        game = await self.game_repo.get_by_slug(slug)
        if not game:
            raise NotFoundError(detail="Game not found")
        return await self.game_repo.toggle_like(game.id, user_id)

    async def toggle_follow(self, slug: str, user_id: UUID) -> bool:
        game = await self.game_repo.get_by_slug(slug)
        if not game:
            raise NotFoundError(detail="Game not found")

        if await self.game_repo.is_following(game.id, user_id):
            await self.game_repo.unfollow_game(game.id, user_id)
            return False

        await self.game_repo.follow_game(game.id, user_id)
        return True

    async def get_featured(self, limit: int = 12) -> list[Game]:
        return await self.game_repo.get_featured(limit)

    async def get_trending(self, limit: int = 12) -> list[Game]:
        return await self.game_repo.get_trending(limit)

    async def get_recent(self, limit: int = 12) -> list[Game]:
        return await self.game_repo.get_recent(limit)

    async def get_recently_updated(self, limit: int = 12) -> list[Game]:
        return await self.game_repo.get_recently_updated(limit)

    async def get_similar(self, slug: str, limit: int = 6) -> list[Game]:
        game = await self.game_repo.get_by_slug(slug)
        if not game:
            raise NotFoundError(detail="Game not found")
        return await self.game_repo.get_similar(game.id, limit)
