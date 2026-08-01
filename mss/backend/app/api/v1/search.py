from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.schemas.game import GameListResponse
from app.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["search"])


class SearchResponse(BaseModel):
    games: list[GameListResponse]
    users: list[dict[str, Any]]
    total: int
    query: str
    page: int
    per_page: int


class SearchSuggestionResponse(BaseModel):
    suggestions: list[str]


def get_search_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SearchService:
    return SearchService(db)


@router.get(
    "",
    response_model=SearchResponse,
    summary="Search games and users",
)
async def search(
    service: Annotated[SearchService, Depends(get_search_service)],
    q: str = Query(min_length=1),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=24, ge=1, le=100),
) -> SearchResponse:
    """Search games and users via Meilisearch."""
    return await service.search(q, page=page, per_page=per_page)


@router.get(
    "/suggestions",
    response_model=SearchSuggestionResponse,
    summary="Get search autocomplete suggestions",
)
async def search_suggestions(
    service: Annotated[SearchService, Depends(get_search_service)],
    q: str = Query(min_length=1),
    limit: int = Query(default=10, ge=1, le=20),
) -> SearchSuggestionResponse:
    """Return autocomplete suggestions for a search query."""
    suggestions = await service.get_suggestions(q, limit=limit)
    return SearchSuggestionResponse(suggestions=suggestions)
