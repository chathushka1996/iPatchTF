from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import PaginationParams, get_current_active_user, get_db, get_pagination
from app.models.user import User
from app.schemas.collection import CollectionResponse
from app.schemas.common import MessageResponse, PaginatedResponse
from app.schemas.game import GameListResponse
from app.schemas.review import ReviewResponse
from app.schemas.user import (
    NotificationPreferencesUpdate,
    PasswordChangeRequest,
    UserBriefResponse,
    UserPublicResponse,
    UserResponse,
    UserUpdate,
)
from app.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["users"])


def get_user_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserService:
    return UserService(db)


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
async def get_current_user_profile(
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[UserService, Depends(get_user_service)],
) -> UserResponse:
    """Return the authenticated user's full profile."""
    return await service.get_profile(current_user)


@router.put(
    "/me",
    response_model=UserResponse,
    summary="Update current user profile",
)
async def update_current_user_profile(
    data: UserUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[UserService, Depends(get_user_service)],
) -> UserResponse:
    """Update the authenticated user's profile fields."""
    return await service.update_profile(current_user, data)


@router.delete(
    "/me",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete current user account",
)
async def delete_current_user_account(
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[UserService, Depends(get_user_service)],
) -> None:
    """Soft-delete the authenticated user's account."""
    await service.delete_account(current_user)


@router.get(
    "/{username}",
    response_model=UserPublicResponse,
    summary="Get public user profile",
)
async def get_user_public_profile(
    username: str,
    service: Annotated[UserService, Depends(get_user_service)],
) -> UserPublicResponse:
    """Return a user's public profile by username."""
    return await service.get_public_profile(username)


@router.get(
    "/{username}/games",
    response_model=PaginatedResponse[GameListResponse],
    summary="List games by user",
)
async def get_user_games(
    username: str,
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[UserService, Depends(get_user_service)],
) -> PaginatedResponse[GameListResponse]:
    """Return paginated games submitted by a user."""
    return await service.get_user_games(username, pagination)


@router.get(
    "/{username}/reviews",
    response_model=PaginatedResponse[ReviewResponse],
    summary="List reviews by user",
)
async def get_user_reviews(
    username: str,
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[UserService, Depends(get_user_service)],
) -> PaginatedResponse[ReviewResponse]:
    """Return paginated reviews written by a user."""
    return await service.get_user_reviews(username, pagination)


@router.get(
    "/{username}/collections",
    response_model=PaginatedResponse[CollectionResponse],
    summary="List public collections by user",
)
async def get_user_collections(
    username: str,
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[UserService, Depends(get_user_service)],
) -> PaginatedResponse[CollectionResponse]:
    """Return paginated public collections owned by a user."""
    return await service.get_user_collections(username, pagination)


@router.post(
    "/{username}/follow",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Follow a user",
)
async def follow_user(
    username: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[UserService, Depends(get_user_service)],
) -> MessageResponse:
    """Follow another user."""
    await service.follow_user(current_user, username)
    return MessageResponse(message="User followed")


@router.delete(
    "/{username}/follow",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Unfollow a user",
)
async def unfollow_user(
    username: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[UserService, Depends(get_user_service)],
) -> None:
    """Unfollow a user."""
    await service.unfollow_user(current_user, username)


@router.get(
    "/{username}/followers",
    response_model=PaginatedResponse[UserBriefResponse],
    summary="List user followers",
)
async def get_user_followers(
    username: str,
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[UserService, Depends(get_user_service)],
) -> PaginatedResponse[UserBriefResponse]:
    """Return paginated followers of a user."""
    return await service.get_followers(username, pagination)


@router.get(
    "/{username}/following",
    response_model=PaginatedResponse[UserBriefResponse],
    summary="List users being followed",
)
async def get_user_following(
    username: str,
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[UserService, Depends(get_user_service)],
) -> PaginatedResponse[UserBriefResponse]:
    """Return paginated list of users followed by this user."""
    return await service.get_following(username, pagination)


@router.put(
    "/me/avatar",
    response_model=UserResponse,
    summary="Upload user avatar",
)
async def upload_avatar(
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[UserService, Depends(get_user_service)],
    file: UploadFile = File(...),
) -> UserResponse:
    """Upload or replace the authenticated user's avatar image."""
    return await service.update_avatar(current_user, file)


@router.put(
    "/me/password",
    response_model=MessageResponse,
    summary="Change password",
)
async def change_password(
    data: PasswordChangeRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[UserService, Depends(get_user_service)],
) -> MessageResponse:
    """Change the authenticated user's password."""
    await service.change_password(current_user, data)
    return MessageResponse(message="Password changed successfully")


@router.put(
    "/me/notifications",
    response_model=MessageResponse,
    summary="Update notification preferences",
)
async def update_notification_preferences(
    data: NotificationPreferencesUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[UserService, Depends(get_user_service)],
) -> MessageResponse:
    """Update the authenticated user's notification preferences."""
    await service.update_notification_preferences(current_user, data.root)
    return MessageResponse(message="Notification preferences updated")
