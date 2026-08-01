from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import ConflictError, ForbiddenError, NotFoundError
from app.models.game import Game
from app.models.review import Review
from app.repositories.game_repo import GameRepository
from app.repositories.review_repo import ReviewRepository


class ReviewService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.review_repo = ReviewRepository(session)
        self.game_repo = GameRepository(session)

    async def _get_game_by_slug(self, slug: str) -> Game:
        game = await self.game_repo.get_by_slug(slug)
        if not game:
            raise NotFoundError(detail="Game not found")
        return game

    async def create_review(
        self, slug: str, data: dict[str, Any], user_id: UUID
    ) -> Review:
        game = await self._get_game_by_slug(slug)
        existing = await self.review_repo.get_user_review_for_game(user_id, game.id)
        if existing:
            raise ConflictError(detail="You have already reviewed this game")

        review = await self.review_repo.create(
            {
                "game_id": game.id,
                "user_id": user_id,
                "score": data["score"],
                "body": data["body"],
                "version_reviewed": data.get("version_reviewed"),
            }
        )

        from app.tasks.cleanup_tasks import recalculate_game_scores

        recalculate_game_scores.delay(str(game.id))
        return review

    async def update_review(
        self, review_id: UUID, data: dict[str, Any], user_id: UUID
    ) -> Review:
        review = await self.review_repo.get_by_id(review_id)
        if not review:
            raise NotFoundError(detail="Review not found")
        if review.user_id != user_id:
            raise ForbiddenError(detail="Not authorized to update this review")

        update_data = {**data, "is_edited": True}
        updated = await self.review_repo.update(review_id, update_data)
        if not updated:
            raise NotFoundError(detail="Review not found")

        from app.tasks.cleanup_tasks import recalculate_game_scores

        recalculate_game_scores.delay(str(review.game_id))
        return updated

    async def delete_review(self, review_id: UUID, user_id: UUID) -> bool:
        review = await self.review_repo.get_by_id(review_id)
        if not review:
            raise NotFoundError(detail="Review not found")
        if review.user_id != user_id:
            raise ForbiddenError(detail="Not authorized to delete this review")

        game_id = review.game_id
        deleted = await self.review_repo.delete(review_id)
        if deleted:
            from app.tasks.cleanup_tasks import recalculate_game_scores

            recalculate_game_scores.delay(str(game_id))
        return deleted

    async def vote_review(
        self, review_id: UUID, user_id: UUID, is_helpful: bool
    ) -> Review:
        review = await self.review_repo.get_by_id(review_id)
        if not review:
            raise NotFoundError(detail="Review not found")
        if review.user_id == user_id:
            raise ForbiddenError(detail="Cannot vote on your own review")

        result = await self.review_repo.vote(review_id, user_id, is_helpful)
        if not result:
            raise NotFoundError(detail="Review not found")
        return result

    async def get_by_game(
        self,
        slug: str,
        skip: int = 0,
        limit: int = 24,
        sort: str = "newest",
    ) -> list[Review]:
        game = await self._get_game_by_slug(slug)
        return await self.review_repo.get_by_game(game.id, skip, limit, sort)

    async def get_recent(self, limit: int = 20) -> list[Review]:
        return await self.review_repo.get_recent(limit)
