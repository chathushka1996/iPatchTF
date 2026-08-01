from typing import Any
from uuid import UUID

from meilisearch_python_sdk import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.game import Game
from app.repositories.game_repo import GameRepository

GAMES_INDEX = "games"


class SearchService:
    def __init__(self, session: AsyncSession | None = None) -> None:
        self.session = session
        self._client = AsyncClient(settings.MEILI_URL, settings.MEILI_MASTER_KEY)

    def _game_document(self, game: Game) -> dict[str, Any]:
        return {
            "id": str(game.id),
            "title": game.title,
            "slug": game.slug,
            "synopsis": game.synopsis or "",
            "engine_id": game.engine_id,
            "author_id": str(game.author_id),
            "rating": game.rating.value if game.rating else None,
            "development_status": (
                game.development_status.value if game.development_status else None
            ),
            "is_free": game.is_free,
            "like_count": game.like_count,
            "average_score": float(game.average_score),
            "view_count": game.view_count,
            "is_featured": game.is_featured,
            "created_at": game.created_at.isoformat() if game.created_at else None,
            "updated_at": game.updated_at.isoformat() if game.updated_at else None,
        }

    async def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        page: int = 1,
        per_page: int = 24,
    ) -> dict[str, Any]:
        filter_parts: list[str] = []
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    filter_parts.append(
                        f"{key} IN [{', '.join(repr(v) for v in value)}]"
                    )
                else:
                    filter_parts.append(f"{key} = {value!r}")

        index = self._client.index(GAMES_INDEX)
        result = await index.search(
            query,
            filter=" AND ".join(filter_parts) if filter_parts else None,
            limit=per_page,
            offset=(page - 1) * per_page,
        )
        return {
            "hits": result.hits,
            "total": result.estimated_total_hits or 0,
            "page": page,
            "per_page": per_page,
        }

    async def get_suggestions(self, query: str, limit: int = 5) -> list[str]:
        index = self._client.index(GAMES_INDEX)
        result = await index.search(query, limit=limit, attributes_to_retrieve=["title"])
        return [hit["title"] for hit in result.hits if "title" in hit]

    async def sync_game(self, game_id: UUID) -> None:
        if not self.session:
            return

        repo = GameRepository(self.session)
        game = await repo.get_by_id(game_id)
        if not game:
            return

        index = self._client.index(GAMES_INDEX)
        await index.add_documents([self._game_document(game)])

    async def delete_game(self, game_id: UUID) -> None:
        index = self._client.index(GAMES_INDEX)
        await index.delete_document(str(game_id))

    async def configure_index(self) -> None:
        index = self._client.index(GAMES_INDEX)
        await index.update_searchable_attributes(
            ["title", "synopsis", "slug"]
        )
        await index.update_filterable_attributes(
            [
                "engine_id",
                "rating",
                "development_status",
                "is_free",
                "is_featured",
                "author_id",
            ]
        )
        await index.update_sortable_attributes(
            ["created_at", "updated_at", "like_count", "average_score", "view_count"]
        )
