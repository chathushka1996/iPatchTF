from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import bleach
import markdown
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import ForbiddenError, NotFoundError
from app.models.forum import ForumCategory, Post, Thread
from app.repositories.forum_repo import ForumRepository
from app.utils.slugify import generate_unique_slug

ALLOWED_TAGS = [
    "p", "br", "strong", "em", "u", "s", "a", "ul", "ol", "li",
    "blockquote", "code", "pre", "h1", "h2", "h3", "h4", "h5", "h6",
    "img", "hr",
]
ALLOWED_ATTRIBUTES = {
    "a": ["href", "title", "rel"],
    "img": ["src", "alt", "title"],
}


class ForumService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.forum_repo = ForumRepository(session)

    def render_markdown(self, text: str) -> str:
        html = markdown.markdown(text, extensions=["fenced_code", "tables", "nl2br"])
        return bleach.clean(
            html,
            tags=ALLOWED_TAGS,
            attributes=ALLOWED_ATTRIBUTES,
            strip=True,
        )

    async def list_categories(self) -> list[ForumCategory]:
        return await self.forum_repo.get_categories()

    async def get_category(self, slug: str) -> ForumCategory:
        category = await self.forum_repo.get_category_by_slug(slug)
        if not category:
            raise NotFoundError(detail="Category not found")
        return category

    async def create_thread(
        self, category_slug: str, data: dict[str, Any], user_id: UUID
    ) -> Thread:
        category = await self.get_category(category_slug)
        slug = await generate_unique_slug(data["title"], Thread, self.session)

        thread = await self.forum_repo.create_thread(
            {
                "forum_category_id": category.id,
                "user_id": user_id,
                "title": data["title"],
                "slug": slug,
                "game_id": data.get("game_id"),
            }
        )

        if data.get("body"):
            await self.forum_repo.create_post(
                {
                    "thread_id": thread.id,
                    "user_id": user_id,
                    "body": data["body"],
                    "body_html": self.render_markdown(data["body"]),
                }
            )

        return thread

    async def get_thread(self, slug: str) -> Thread:
        thread = await self.forum_repo.get_thread_by_slug(slug)
        if not thread:
            raise NotFoundError(detail="Thread not found")
        await self.forum_repo.increment_view_count(thread.id)
        return thread

    async def create_post(
        self, thread_slug: str, data: dict[str, Any], user_id: UUID
    ) -> Post:
        thread = await self.get_thread(thread_slug)
        if thread.is_locked:
            raise ForbiddenError(detail="Thread is locked")

        post = await self.forum_repo.create_post(
            {
                "thread_id": thread.id,
                "user_id": user_id,
                "body": data["body"],
                "body_html": self.render_markdown(data["body"]),
                "parent_id": data.get("parent_id"),
            }
        )

        thread.post_count += 1
        thread.last_post_at = datetime.now(UTC)
        await self.session.flush()
        return post

    async def update_post(
        self, post_id: UUID, data: dict[str, Any], user_id: UUID
    ) -> Post:
        post = await self.forum_repo.get_post_by_id(post_id)
        if not post:
            raise NotFoundError(detail="Post not found")
        if post.user_id != user_id:
            raise ForbiddenError(detail="Not authorized to update this post")

        update_data = {
            "body": data["body"],
            "body_html": self.render_markdown(data["body"]),
            "is_edited": True,
            "edited_at": datetime.now(UTC),
        }
        updated = await self.forum_repo.update_post(post_id, update_data)
        if not updated:
            raise NotFoundError(detail="Post not found")
        return updated

    async def delete_post(self, post_id: UUID, user_id: UUID) -> bool:
        post = await self.forum_repo.get_post_by_id(post_id)
        if not post:
            raise NotFoundError(detail="Post not found")
        if post.user_id != user_id:
            raise ForbiddenError(detail="Not authorized to delete this post")
        return await self.forum_repo.delete_post(post_id)

    async def lock_thread(self, slug: str) -> Thread:
        thread = await self.forum_repo.get_thread_by_slug(slug)
        if not thread:
            raise NotFoundError(detail="Thread not found")
        updated = await self.forum_repo.update_thread(thread.id, {"is_locked": True})
        if not updated:
            raise NotFoundError(detail="Thread not found")
        return updated

    async def pin_thread(self, slug: str) -> Thread:
        thread = await self.forum_repo.get_thread_by_slug(slug)
        if not thread:
            raise NotFoundError(detail="Thread not found")
        updated = await self.forum_repo.update_thread(thread.id, {"is_pinned": True})
        if not updated:
            raise NotFoundError(detail="Thread not found")
        return updated
