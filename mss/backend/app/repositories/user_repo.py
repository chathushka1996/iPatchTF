from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.user import Follow, User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(User, session)

    async def get_by_email(self, email: str) -> User | None:
        result = await self.session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> User | None:
        result = await self.session.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def get_followers(
        self, user_id: UUID, skip: int = 0, limit: int = 50
    ) -> list[User]:
        result = await self.session.execute(
            select(User)
            .join(Follow, Follow.follower_id == User.id)
            .where(Follow.following_id == user_id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_following(
        self, user_id: UUID, skip: int = 0, limit: int = 50
    ) -> list[User]:
        result = await self.session.execute(
            select(User)
            .join(Follow, Follow.following_id == User.id)
            .where(Follow.follower_id == user_id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def follow(self, follower_id: UUID, following_id: UUID) -> Follow:
        follow = Follow(follower_id=follower_id, following_id=following_id)
        self.session.add(follow)
        await self.session.flush()
        return follow

    async def unfollow(self, follower_id: UUID, following_id: UUID) -> bool:
        result = await self.session.execute(
            delete(Follow).where(
                Follow.follower_id == follower_id,
                Follow.following_id == following_id,
            )
        )
        await self.session.flush()
        return result.rowcount > 0

    async def update_last_login(self, user_id: UUID) -> None:
        user = await self.get_by_id(user_id)
        if user:
            user.last_login_at = datetime.now(UTC)
            await self.session.flush()

    async def get_with_profile(self, user_id: UUID) -> User | None:
        result = await self.session.execute(
            select(User)
            .options(selectinload(User.profile))
            .where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def follower_count(self, user_id: UUID) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(Follow)
            .where(Follow.following_id == user_id)
        )
        return result.scalar_one()

    async def following_count(self, user_id: UUID) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(Follow)
            .where(Follow.follower_id == user_id)
        )
        return result.scalar_one()
