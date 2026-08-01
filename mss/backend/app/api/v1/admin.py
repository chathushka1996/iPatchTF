from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import PaginationParams, get_db, get_pagination, require_role
from app.models.base import UserRole
from app.models.user import User
from app.schemas.admin import (
    AdminDashboardResponse,
    EngineCreate,
    ReportUpdateRequest,
    RoleChangeRequest,
    TagCreate,
)
from app.schemas.common import MessageResponse, PaginatedResponse
from app.schemas.forum import ForumCategoryResponse
from app.schemas.game import EngineResponse, GameListResponse, TagResponse
from app.schemas.user import UserResponse
from app.services.moderation_service import ModerationService

router = APIRouter(prefix="/admin", tags=["admin"])


class ReportResponse(BaseModel):
    id: UUID
    reporter_id: UUID
    reason: str
    description: str | None = None
    target_type: str
    target_id: UUID
    status: str
    resolution_note: str | None = None
    created_at: datetime
    resolved_at: datetime | None = None


class AuditLogResponse(BaseModel):
    id: UUID
    user_id: UUID
    action: str
    target_type: str
    target_id: UUID
    metadata: dict[str, Any] = Field(default_factory=dict)
    ip_address: str | None = None
    created_at: datetime


class ForumCategoryCreate(BaseModel):
    name: str = Field(max_length=100)
    slug: str | None = Field(default=None, max_length=120)
    description: str | None = None
    parent_id: int | None = None
    sort_order: int = 0


class ForumCategoryUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    slug: str | None = Field(default=None, max_length=120)
    description: str | None = None
    parent_id: int | None = None
    sort_order: int | None = None
    is_locked: bool | None = None


class TagUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    slug: str | None = Field(default=None, max_length=120)
    category: str | None = None
    description: str | None = None


class EngineUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    slug: str | None = Field(default=None, max_length=120)


class RejectGameRequest(BaseModel):
    reason: str = Field(min_length=1)


def get_moderation_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModerationService:
    return ModerationService(db)


@router.get(
    "/dashboard",
    response_model=AdminDashboardResponse,
    summary="Get admin dashboard stats",
)
async def get_dashboard(
    _admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> AdminDashboardResponse:
    """Return aggregate platform statistics."""
    return await service.get_dashboard_stats()


@router.get(
    "/users",
    response_model=PaginatedResponse[UserResponse],
    summary="List users",
)
async def list_users(
    _admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
    q: str | None = None,
    role: str | None = None,
    is_active: bool | None = None,
) -> PaginatedResponse[UserResponse]:
    """Return a paginated list of users with optional filters."""
    return await service.list_users(pagination, q=q, role=role, is_active=is_active)


@router.put(
    "/users/{user_id}/role",
    response_model=UserResponse,
    summary="Change user role",
)
async def change_user_role(
    user_id: UUID,
    data: RoleChangeRequest,
    admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> UserResponse:
    """Change a user's role."""
    return await service.change_user_role(user_id, data.role, admin)


@router.post(
    "/users/{user_id}/ban",
    response_model=MessageResponse,
    summary="Ban a user",
)
async def ban_user(
    user_id: UUID,
    admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> MessageResponse:
    """Ban a user account."""
    await service.ban_user(user_id, admin)
    return MessageResponse(message="User banned")


@router.post(
    "/users/{user_id}/unban",
    response_model=MessageResponse,
    summary="Unban a user",
)
async def unban_user(
    user_id: UUID,
    admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> MessageResponse:
    """Restore a banned user account."""
    await service.unban_user(user_id, admin)
    return MessageResponse(message="User unbanned")


@router.get(
    "/games/pending",
    response_model=PaginatedResponse[GameListResponse],
    summary="List pending games",
)
async def list_pending_games(
    _admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> PaginatedResponse[GameListResponse]:
    """Return games awaiting approval."""
    return await service.list_pending_games(pagination)


@router.post(
    "/games/{game_id}/approve",
    response_model=MessageResponse,
    summary="Approve a pending game",
)
async def approve_game(
    game_id: UUID,
    admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> MessageResponse:
    """Approve a pending game submission."""
    await service.approve_game(game_id, admin)
    return MessageResponse(message="Game approved")


@router.post(
    "/games/{game_id}/reject",
    response_model=MessageResponse,
    summary="Reject a pending game",
)
async def reject_game(
    game_id: UUID,
    data: RejectGameRequest,
    admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> MessageResponse:
    """Reject a pending game submission."""
    await service.reject_game(game_id, admin, data.reason)
    return MessageResponse(message="Game rejected")


@router.get(
    "/reports",
    response_model=PaginatedResponse[ReportResponse],
    summary="List moderation reports",
)
async def list_reports(
    _admin: Annotated[User, Depends(require_role(UserRole.MODERATOR, UserRole.ADMIN))],
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
    status_filter: str | None = Query(default=None, alias="status"),
) -> PaginatedResponse[ReportResponse]:
    """Return the moderation report queue."""
    return await service.list_reports(pagination, status=status_filter)


@router.put(
    "/reports/{report_id}",
    response_model=ReportResponse,
    summary="Resolve or dismiss a report",
)
async def update_report(
    report_id: UUID,
    data: ReportUpdateRequest,
    moderator: Annotated[User, Depends(require_role(UserRole.MODERATOR, UserRole.ADMIN))],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> ReportResponse:
    """Resolve or dismiss a moderation report."""
    return await service.update_report(report_id, moderator, data)


@router.get(
    "/audit-log",
    response_model=PaginatedResponse[AuditLogResponse],
    summary="Get audit log",
)
async def get_audit_log(
    _admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> PaginatedResponse[AuditLogResponse]:
    """Return the system audit trail."""
    return await service.get_audit_log(pagination)


@router.delete(
    "/reviews/{review_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Force delete a review",
)
async def force_delete_review(
    review_id: UUID,
    admin: Annotated[User, Depends(require_role(UserRole.MODERATOR, UserRole.ADMIN))],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> None:
    """Force-delete a review. Moderator only."""
    await service.force_delete_review(review_id, admin)


@router.delete(
    "/posts/{post_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Force delete a post",
)
async def force_delete_post(
    post_id: UUID,
    admin: Annotated[User, Depends(require_role(UserRole.MODERATOR, UserRole.ADMIN))],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> None:
    """Force-delete a forum post. Moderator only."""
    await service.force_delete_post(post_id, admin)


@router.post(
    "/tags",
    response_model=TagResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a tag",
)
async def create_tag(
    data: TagCreate,
    admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> TagResponse:
    """Create a new game tag."""
    return await service.create_tag(data, admin)


@router.put(
    "/tags/{tag_id}",
    response_model=TagResponse,
    summary="Edit a tag",
)
async def update_tag(
    tag_id: int,
    data: TagUpdate,
    admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> TagResponse:
    """Update an existing game tag."""
    return await service.update_tag(tag_id, data, admin)


@router.post(
    "/engines",
    response_model=EngineResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an engine",
)
async def create_engine(
    data: EngineCreate,
    admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> EngineResponse:
    """Create a new game engine entry."""
    return await service.create_engine(data, admin)


@router.put(
    "/engines/{engine_id}",
    response_model=EngineResponse,
    summary="Edit an engine",
)
async def update_engine(
    engine_id: int,
    data: EngineUpdate,
    admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> EngineResponse:
    """Update an existing game engine entry."""
    return await service.update_engine(engine_id, data, admin)


@router.post(
    "/forum-categories",
    response_model=ForumCategoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a forum category",
)
async def create_forum_category(
    data: ForumCategoryCreate,
    admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> ForumCategoryResponse:
    """Create a new forum category."""
    return await service.create_forum_category(data, admin)


@router.put(
    "/forum-categories/{category_id}",
    response_model=ForumCategoryResponse,
    summary="Edit a forum category",
)
async def update_forum_category(
    category_id: int,
    data: ForumCategoryUpdate,
    admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    service: Annotated[ModerationService, Depends(get_moderation_service)],
) -> ForumCategoryResponse:
    """Update an existing forum category."""
    return await service.update_forum_category(category_id, data, admin)
