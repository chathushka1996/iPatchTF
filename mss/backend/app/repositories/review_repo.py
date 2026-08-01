from uuid import UUID

from sqlalchemy import delete, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.review import Review, ReviewVote
from app.repositories.base import BaseRepository


class ReviewRepository(BaseRepository[Review]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Review, session)

    async def get_by_game(
        self,
        game_id: UUID,
        skip: int = 0,
        limit: int = 24,
        sort: str = "newest",
    ) -> list[Review]:
        query = select(Review).where(Review.game_id == game_id)
        sort_map = {
            "newest": desc(Review.created_at),
            "oldest": Review.created_at.asc(),
            "helpful": desc(Review.helpful_count),
            "score_high": desc(Review.score),
            "score_low": Review.score.asc(),
        }
        query = query.order_by(sort_map.get(sort, desc(Review.created_at)))
        result = await self.session.execute(query.offset(skip).limit(limit))
        return list(result.scalars().all())

    async def get_by_user(
        self, user_id: UUID, skip: int = 0, limit: int = 24
    ) -> list[Review]:
        result = await self.session.execute(
            select(Review)
            .where(Review.user_id == user_id)
            .order_by(desc(Review.created_at))
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_user_review_for_game(
        self, user_id: UUID, game_id: UUID
    ) -> Review | None:
        result = await self.session.execute(
            select(Review).where(
                Review.user_id == user_id,
                Review.game_id == game_id,
            )
        )
        return result.scalar_one_or_none()

    async def vote(
        self, review_id: UUID, user_id: UUID, is_helpful: bool
    ) -> Review | None:
        result = await self.session.execute(
            select(ReviewVote).where(
                ReviewVote.review_id == review_id,
                ReviewVote.user_id == user_id,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            if existing.is_helpful == is_helpful:
                await self.session.execute(
                    delete(ReviewVote).where(
                        ReviewVote.review_id == review_id,
                        ReviewVote.user_id == user_id,
                    )
                )
                field = "helpful_count" if is_helpful else "not_helpful_count"
                await self.session.execute(
                    update(Review)
                    .where(Review.id == review_id)
                    .values(**{field: getattr(Review, field) - 1})
                )
            else:
                existing.is_helpful = is_helpful
                await self.session.execute(
                    update(Review)
                    .where(Review.id == review_id)
                    .values(
                        helpful_count=Review.helpful_count + (1 if is_helpful else -1),
                        not_helpful_count=Review.not_helpful_count + (-1 if is_helpful else 1),
                    )
                )
        else:
            self.session.add(
                ReviewVote(review_id=review_id, user_id=user_id, is_helpful=is_helpful)
            )
            field = "helpful_count" if is_helpful else "not_helpful_count"
            await self.session.execute(
                update(Review)
                .where(Review.id == review_id)
                .values(**{field: getattr(Review, field) + 1})
            )

        await self.session.flush()
        return await self.get_by_id(review_id)

    async def get_recent(self, limit: int = 20) -> list[Review]:
        result = await self.session.execute(
            select(Review).order_by(desc(Review.created_at)).limit(limit)
        )
        return list(result.scalars().all())

    async def count_by_game(self, game_id: UUID) -> int:
        result = await self.session.execute(
            select(func.count()).select_from(Review).where(Review.game_id == game_id)
        )
        return result.scalar_one()

    async def average_score_for_game(self, game_id: UUID) -> float:
        result = await self.session.execute(
            select(func.avg(Review.score)).where(Review.game_id == game_id)
        )
        avg = result.scalar_one()
        return float(avg) if avg is not None else 0.0
