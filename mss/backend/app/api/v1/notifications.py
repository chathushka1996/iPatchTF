from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import PaginationParams, get_current_active_user, get_db, get_pagination
from app.models.user import User
from app.schemas.common import MessageResponse, PaginatedResponse
from app.schemas.notification import (
    MarkReadRequest,
    NotificationResponse,
    UnreadCountResponse,
)
from app.services.notification_service import NotificationService

router = APIRouter(prefix="/notifications", tags=["notifications"])


def get_notification_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> NotificationService:
    return NotificationService(db)


@router.get(
    "",
    response_model=PaginatedResponse[NotificationResponse],
    summary="List notifications",
)
async def list_notifications(
    current_user: Annotated[User, Depends(get_current_active_user)],
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[NotificationService, Depends(get_notification_service)],
) -> PaginatedResponse[NotificationResponse]:
    """Return paginated notifications for the authenticated user."""
    return await service.list_notifications(current_user, pagination)


@router.get(
    "/unread-count",
    response_model=UnreadCountResponse,
    summary="Get unread notification count",
)
async def get_unread_count(
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[NotificationService, Depends(get_notification_service)],
) -> UnreadCountResponse:
    """Return the number of unread notifications."""
    return await service.get_unread_count(current_user)


@router.post(
    "/mark-read",
    response_model=MessageResponse,
    summary="Mark notifications as read",
)
async def mark_notifications_read(
    data: MarkReadRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[NotificationService, Depends(get_notification_service)],
) -> MessageResponse:
    """Mark specific notifications or all notifications as read."""
    await service.mark_read(current_user, data.notification_ids)
    return MessageResponse(message="Notifications marked as read")


@router.delete(
    "/{notification_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a notification",
)
async def delete_notification(
    notification_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[NotificationService, Depends(get_notification_service)],
) -> None:
    """Delete a notification owned by the authenticated user."""
    await service.delete_notification(current_user, notification_id)
