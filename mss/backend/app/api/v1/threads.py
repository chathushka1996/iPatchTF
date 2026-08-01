from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_active_user, get_db, require_role
from app.models.base import UserRole
from app.models.user import User
from app.schemas.common import MessageResponse
from app.schemas.forum import (
    PostCreate,
    PostResponse,
    PostUpdate,
    ThreadCreate,
    ThreadDetailResponse,
    ThreadResponse,
)
from app.services.forum_service import ForumService

router = APIRouter(tags=["threads"])


def get_forum_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ForumService:
    return ForumService(db)


@router.post(
    "/forums/{slug}/threads",
    response_model=ThreadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a forum thread",
)
async def create_thread(
    slug: str,
    data: ThreadCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[ForumService, Depends(get_forum_service)],
) -> ThreadResponse:
    """Create a new thread in a forum category."""
    return await service.create_thread(slug, current_user, data)


@router.get(
    "/threads/{slug}",
    response_model=ThreadDetailResponse,
    summary="Get thread detail with posts",
)
async def get_thread(
    slug: str,
    service: Annotated[ForumService, Depends(get_forum_service)],
) -> ThreadDetailResponse:
    """Return a thread and its posts."""
    return await service.get_thread_detail(slug)


@router.post(
    "/threads/{slug}/posts",
    response_model=PostResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Reply to a thread",
)
async def create_post(
    slug: str,
    data: PostCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[ForumService, Depends(get_forum_service)],
) -> PostResponse:
    """Post a reply in a thread."""
    return await service.create_post(slug, current_user, data)


@router.put(
    "/posts/{post_id}",
    response_model=PostResponse,
    summary="Edit a post",
)
async def update_post(
    post_id: UUID,
    data: PostUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[ForumService, Depends(get_forum_service)],
) -> PostResponse:
    """Edit a post owned by the authenticated user."""
    return await service.update_post(post_id, current_user, data)


@router.delete(
    "/posts/{post_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a post",
)
async def delete_post(
    post_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[ForumService, Depends(get_forum_service)],
) -> None:
    """Delete a post owned by the authenticated user or a moderator."""
    await service.delete_post(post_id, current_user)


@router.post(
    "/threads/{slug}/lock",
    response_model=MessageResponse,
    summary="Lock a thread",
)
async def lock_thread(
    slug: str,
    current_user: Annotated[User, Depends(require_role(UserRole.MODERATOR, UserRole.ADMIN))],
    service: Annotated[ForumService, Depends(get_forum_service)],
) -> MessageResponse:
    """Lock a thread to prevent new replies. Moderator only."""
    await service.lock_thread(slug, current_user)
    return MessageResponse(message="Thread locked")


@router.post(
    "/threads/{slug}/pin",
    response_model=MessageResponse,
    summary="Pin a thread",
)
async def pin_thread(
    slug: str,
    current_user: Annotated[User, Depends(require_role(UserRole.MODERATOR, UserRole.ADMIN))],
    service: Annotated[ForumService, Depends(get_forum_service)],
) -> MessageResponse:
    """Pin a thread to the top of its category. Moderator only."""
    await service.pin_thread(slug, current_user)
    return MessageResponse(message="Thread pinned")
