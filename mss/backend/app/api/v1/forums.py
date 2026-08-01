from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import PaginationParams, get_db, get_pagination
from app.schemas.common import PaginatedResponse
from app.schemas.forum import ForumCategoryResponse, ThreadResponse
from app.services.forum_service import ForumService

router = APIRouter(prefix="/forums", tags=["forums"])


def get_forum_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ForumService:
    return ForumService(db)


@router.get(
    "",
    response_model=list[ForumCategoryResponse],
    summary="List forum categories",
)
async def list_forum_categories(
    service: Annotated[ForumService, Depends(get_forum_service)],
) -> list[ForumCategoryResponse]:
    """Return all forum categories with subcategories."""
    return await service.list_categories()


@router.get(
    "/{slug}",
    response_model=PaginatedResponse[ThreadResponse],
    summary="List threads in a forum category",
)
async def list_forum_threads(
    slug: str,
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[ForumService, Depends(get_forum_service)],
) -> PaginatedResponse[ThreadResponse]:
    """Return paginated threads in a forum category."""
    return await service.list_threads(slug, pagination)
