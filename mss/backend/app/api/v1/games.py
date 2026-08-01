from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import PaginationParams, get_current_active_user, get_db, get_pagination
from app.models.base import UserRole
from app.models.user import User
from app.schemas.common import MessageResponse, PaginatedResponse
from app.schemas.game import (
    GameCreate,
    GameListResponse,
    GameResponse,
    GameSearchParams,
    GameSortOption,
    GameUpdate,
    GameVersionCreate,
    GameVersionResponse,
    ScreenshotResponse,
)
from app.services.game_service import GameService

router = APIRouter(prefix="/games", tags=["games"])


class ScreenshotCreate(BaseModel):
    object_key: str = Field(min_length=1, max_length=500)
    caption: str | None = Field(default=None, max_length=500)


class GameVersionUpdate(BaseModel):
    version_string: str | None = Field(default=None, max_length=50)
    changelog: str | None = None


class LikeToggleResponse(BaseModel):
    liked: bool
    like_count: int


def get_game_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> GameService:
    return GameService(db)


def require_admin(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> User:
    if current_user.role not in (UserRole.ADMIN, UserRole.MODERATOR):
        from app.exceptions import ForbiddenError

        raise ForbiddenError(detail="Admin access required")
    return current_user


@router.get(
    "",
    response_model=PaginatedResponse[GameListResponse],
    summary="List games with filters",
)
async def list_games(
    service: Annotated[GameService, Depends(get_game_service)],
    q: str | None = None,
    engine: list[str] | None = Query(default=None),
    status_filter: list[str] | None = Query(default=None, alias="status"),
    genre: list[str] | None = Query(default=None),
    adult_theme: list[str] | None = Query(default=None),
    transformation: list[str] | None = Query(default=None),
    multimedia: list[str] | None = Query(default=None),
    content_warning: list[str] | None = Query(default=None),
    rating: list[str] | None = Query(default=None),
    pc_gender: list[str] | None = Query(default=None),
    author: str | None = None,
    has_play_online: bool | None = None,
    min_likes: int | None = Query(default=None, ge=0),
    sort: GameSortOption = "newest",
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=24, ge=1, le=100),
) -> PaginatedResponse[GameListResponse]:
    """List and filter games with pagination."""
    params = GameSearchParams(
        q=q,
        engine=engine,
        status=status_filter,
        genre=genre,
        adult_theme=adult_theme,
        transformation=transformation,
        multimedia=multimedia,
        content_warning=content_warning,
        rating=rating,
        pc_gender=pc_gender,
        author=author,
        has_play_online=has_play_online,
        min_likes=min_likes,
        sort=sort,
        page=page,
        per_page=per_page,
    )
    return await service.list_games(params)


@router.get(
    "/featured",
    response_model=list[GameListResponse],
    summary="Get featured games",
)
async def get_featured_games(
    service: Annotated[GameService, Depends(get_game_service)],
) -> list[GameListResponse]:
    """Return curated featured games."""
    return await service.get_featured_games()


@router.get(
    "/trending",
    response_model=list[GameListResponse],
    summary="Get trending games",
)
async def get_trending_games(
    service: Annotated[GameService, Depends(get_game_service)],
) -> list[GameListResponse]:
    """Return games trending this week."""
    return await service.get_trending_games()


@router.get(
    "/recent",
    response_model=PaginatedResponse[GameListResponse],
    summary="Get recently submitted games",
)
async def get_recent_games(
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[GameService, Depends(get_game_service)],
) -> PaginatedResponse[GameListResponse]:
    """Return recently submitted games."""
    return await service.get_recent_games(pagination)


@router.get(
    "/recently-updated",
    response_model=PaginatedResponse[GameListResponse],
    summary="Get recently updated games",
)
async def get_recently_updated_games(
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[GameService, Depends(get_game_service)],
) -> PaginatedResponse[GameListResponse]:
    """Return games with recent updates."""
    return await service.get_recently_updated_games(pagination)


@router.get(
    "/{slug}",
    response_model=GameResponse,
    summary="Get game detail",
)
async def get_game(
    slug: str,
    service: Annotated[GameService, Depends(get_game_service)],
) -> GameResponse:
    """Return full details for a game by slug."""
    return await service.get_game_by_slug(slug)


@router.post(
    "",
    response_model=GameResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a new game",
)
async def create_game(
    data: GameCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[GameService, Depends(get_game_service)],
) -> GameResponse:
    """Submit a new game for review."""
    return await service.create_game(current_user, data)


@router.put(
    "/{slug}",
    response_model=GameResponse,
    summary="Update a game",
)
async def update_game(
    slug: str,
    data: GameUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[GameService, Depends(get_game_service)],
) -> GameResponse:
    """Update game metadata. Requires author or admin privileges."""
    return await service.update_game(slug, current_user, data)


@router.delete(
    "/{slug}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft delete a game",
)
async def delete_game(
    slug: str,
    current_user: Annotated[User, Depends(require_admin)],
    service: Annotated[GameService, Depends(get_game_service)],
) -> None:
    """Soft-delete a game. Admin only."""
    await service.delete_game(slug, current_user)


@router.post(
    "/{slug}/versions",
    response_model=GameVersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a game version",
)
async def add_game_version(
    slug: str,
    data: GameVersionCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[GameService, Depends(get_game_service)],
) -> GameVersionResponse:
    """Add a new version to a game."""
    return await service.add_version(slug, current_user, data)


@router.get(
    "/{slug}/versions",
    response_model=list[GameVersionResponse],
    summary="Get game version history",
)
async def get_game_versions(
    slug: str,
    service: Annotated[GameService, Depends(get_game_service)],
) -> list[GameVersionResponse]:
    """Return version history for a game."""
    return await service.get_versions(slug)


@router.put(
    "/{slug}/versions/{version_id}",
    response_model=GameVersionResponse,
    summary="Update a game version",
)
async def update_game_version(
    slug: str,
    version_id: UUID,
    data: GameVersionUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[GameService, Depends(get_game_service)],
) -> GameVersionResponse:
    """Edit version metadata for a game."""
    return await service.update_version(slug, version_id, current_user, data)


@router.post(
    "/{slug}/screenshots",
    response_model=ScreenshotResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a screenshot",
)
async def add_screenshot(
    slug: str,
    data: ScreenshotCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[GameService, Depends(get_game_service)],
) -> ScreenshotResponse:
    """Register a screenshot after uploading via presigned URL."""
    return await service.add_screenshot(slug, current_user, data.object_key, data.caption)


@router.delete(
    "/{slug}/screenshots/{screenshot_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a screenshot",
)
async def delete_screenshot(
    slug: str,
    screenshot_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[GameService, Depends(get_game_service)],
) -> None:
    """Remove a screenshot from a game."""
    await service.delete_screenshot(slug, screenshot_id, current_user)


@router.post(
    "/{slug}/like",
    response_model=LikeToggleResponse,
    summary="Toggle game like",
)
async def toggle_game_like(
    slug: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[GameService, Depends(get_game_service)],
) -> LikeToggleResponse:
    """Like or unlike a game."""
    return await service.toggle_like(slug, current_user)


@router.get(
    "/{slug}/similar",
    response_model=list[GameListResponse],
    summary="Get similar games",
)
async def get_similar_games(
    slug: str,
    service: Annotated[GameService, Depends(get_game_service)],
) -> list[GameListResponse]:
    """Return games similar to the given game."""
    return await service.get_similar_games(slug)


@router.post(
    "/{slug}/follow",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Follow a game",
)
async def follow_game(
    slug: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[GameService, Depends(get_game_service)],
) -> MessageResponse:
    """Follow a game for update notifications."""
    await service.follow_game(slug, current_user)
    return MessageResponse(message="Game followed")


@router.delete(
    "/{slug}/follow",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Unfollow a game",
)
async def unfollow_game(
    slug: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[GameService, Depends(get_game_service)],
) -> None:
    """Unfollow a game."""
    await service.unfollow_game(slug, current_user)
