from typing import Any
from uuid import UUID

from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.forum import ForumCategory, Post, Thread


class ForumRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_categories(self) -> list[ForumCategory]:
        result = await self.session.execute(
            select(ForumCategory)
            .where(ForumCategory.parent_id.is_(None))
            .order_by(ForumCategory.sort_order.nulls_last(), ForumCategory.name)
        )
        return list(result.scalars().all())

    async def get_category_by_slug(self, slug: str) -> ForumCategory | None:
        result = await self.session.execute(
            select(ForumCategory).where(ForumCategory.slug == slug)
        )
        return result.scalar_one_or_none()

    async def get_threads(
        self, category_id: int, skip: int = 0, limit: int = 24
    ) -> list[Thread]:
        result = await self.session.execute(
            select(Thread)
            .where(Thread.forum_category_id == category_id)
            .order_by(desc(Thread.is_pinned), desc(Thread.last_post_at))
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_thread_by_slug(self, slug: str) -> Thread | None:
        result = await self.session.execute(
            select(Thread)
            .options(selectinload(Thread.posts))
            .where(Thread.slug == slug)
        )
        return result.scalar_one_or_none()

    async def create_thread(self, data: dict[str, Any]) -> Thread:
        thread = Thread(**data)
        self.session.add(thread)
        await self.session.flush()
        await self.session.refresh(thread)
        return thread

    async def create_post(self, data: dict[str, Any]) -> Post:
        post = Post(**data)
        self.session.add(post)
        await self.session.flush()
        await self.session.refresh(post)
        return post

    async def get_posts(
        self, thread_id: UUID, skip: int = 0, limit: int = 50
    ) -> list[Post]:
        result = await self.session.execute(
            select(Post)
            .where(Post.thread_id == thread_id, Post.parent_id.is_(None))
            .order_by(Post.created_at)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_post_by_id(self, post_id: UUID) -> Post | None:
        return await self.session.get(Post, post_id)

    async def update_post(self, post_id: UUID, data: dict[str, Any]) -> Post | None:
        post = await self.get_post_by_id(post_id)
        if not post:
            return None
        for key, value in data.items():
            setattr(post, key, value)
        await self.session.flush()
        await self.session.refresh(post)
        return post

    async def delete_post(self, post_id: UUID) -> bool:
        post = await self.get_post_by_id(post_id)
        if not post:
            return False
        await self.session.delete(post)
        await self.session.flush()
        return True

    async def increment_view_count(self, thread_id: UUID) -> None:
        await self.session.execute(
            update(Thread)
            .where(Thread.id == thread_id)
            .values(view_count=Thread.view_count + 1)
        )
        await self.session.flush()

    async def update_thread(self, thread_id: UUID, data: dict[str, Any]) -> Thread | None:
        thread = await self.session.get(Thread, thread_id)
        if not thread:
            return None
        for key, value in data.items():
            setattr(thread, key, value)
        await self.session.flush()
        await self.session.refresh(thread)
        return thread
