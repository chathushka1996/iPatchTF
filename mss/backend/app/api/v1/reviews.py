from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import PaginationParams, get_current_active_user, get_db, get_pagination
from app.models.user import User
from app.schemas.common import PaginatedResponse
from app.schemas.review import (
    ReviewCreate,
    ReviewResponse,
    ReviewUpdate,
    ReviewVoteRequest,
)
from app.services.review_service import ReviewService

router = APIRouter(tags=["reviews"])


def get_review_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ReviewService:
    return ReviewService(db)


@router.get(
    "/games/{slug}/reviews",
    response_model=PaginatedResponse[ReviewResponse],
    summary="List reviews for a game",
)
async def list_game_reviews(
    slug: str,
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[ReviewService, Depends(get_review_service)],
) -> PaginatedResponse[ReviewResponse]:
    """Return paginated reviews for a game."""
    return await service.list_game_reviews(slug, pagination)


@router.post(
    "/games/{slug}/reviews",
    response_model=ReviewResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a review",
)
async def create_review(
    slug: str,
    data: ReviewCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[ReviewService, Depends(get_review_service)],
) -> ReviewResponse:
    """Submit a review for a game. One review per user per game."""
    return await service.create_review(slug, current_user, data)


@router.get(
    "/reviews/recent",
    response_model=PaginatedResponse[ReviewResponse],
    summary="Get recent reviews globally",
)
async def list_recent_reviews(
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[ReviewService, Depends(get_review_service)],
) -> PaginatedResponse[ReviewResponse]:
    """Return a global feed of recent reviews."""
    return await service.list_recent_reviews(pagination)


@router.put(
    "/reviews/{review_id}",
    response_model=ReviewResponse,
    summary="Edit own review",
)
async def update_review(
    review_id: UUID,
    data: ReviewUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[ReviewService, Depends(get_review_service)],
) -> ReviewResponse:
    """Update a review owned by the authenticated user."""
    return await service.update_review(review_id, current_user, data)


@router.delete(
    "/reviews/{review_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete own review",
)
async def delete_review(
    review_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[ReviewService, Depends(get_review_service)],
) -> None:
    """Delete a review owned by the authenticated user."""
    await service.delete_review(review_id, current_user)


@router.post(
    "/reviews/{review_id}/vote",
    response_model=ReviewResponse,
    summary="Vote on a review",
)
async def vote_review(
    review_id: UUID,
    data: ReviewVoteRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[ReviewService, Depends(get_review_service)],
) -> ReviewResponse:
    """Vote a review as helpful or not helpful."""
    return await service.vote_review(review_id, current_user, data.is_helpful)
