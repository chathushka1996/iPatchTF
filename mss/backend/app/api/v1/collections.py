from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import PaginationParams, get_current_active_user, get_db, get_pagination
from app.models.user import User
from app.schemas.collection import (
    CollectionCreate,
    CollectionDetailResponse,
    CollectionGameAdd,
    CollectionResponse,
    CollectionUpdate,
    ReorderRequest,
)
from app.schemas.common import MessageResponse, PaginatedResponse
from app.services.collection_service import CollectionService

router = APIRouter(prefix="/collections", tags=["collections"])


def get_collection_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CollectionService:
    return CollectionService(db)


@router.get(
    "",
    response_model=PaginatedResponse[CollectionResponse],
    summary="List public collections",
)
async def list_collections(
    pagination: Annotated[PaginationParams, Depends(get_pagination)],
    service: Annotated[CollectionService, Depends(get_collection_service)],
) -> PaginatedResponse[CollectionResponse]:
    """Return a feed of public collections."""
    return await service.list_public_collections(pagination)


@router.post(
    "",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a collection",
)
async def create_collection(
    data: CollectionCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[CollectionService, Depends(get_collection_service)],
) -> CollectionResponse:
    """Create a new game collection."""
    return await service.create_collection(current_user, data)


@router.get(
    "/{collection_id}",
    response_model=CollectionDetailResponse,
    summary="Get collection detail",
)
async def get_collection(
    collection_id: UUID,
    service: Annotated[CollectionService, Depends(get_collection_service)],
) -> CollectionDetailResponse:
    """Return a collection with its games."""
    return await service.get_collection(collection_id)


@router.put(
    "/{collection_id}",
    response_model=CollectionResponse,
    summary="Update a collection",
)
async def update_collection(
    collection_id: UUID,
    data: CollectionUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[CollectionService, Depends(get_collection_service)],
) -> CollectionResponse:
    """Update collection name, description, or visibility."""
    return await service.update_collection(collection_id, current_user, data)


@router.delete(
    "/{collection_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a collection",
)
async def delete_collection(
    collection_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[CollectionService, Depends(get_collection_service)],
) -> None:
    """Delete a collection owned by the authenticated user."""
    await service.delete_collection(collection_id, current_user)


@router.post(
    "/{collection_id}/games",
    response_model=CollectionDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a game to a collection",
)
async def add_game_to_collection(
    collection_id: UUID,
    data: CollectionGameAdd,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[CollectionService, Depends(get_collection_service)],
) -> CollectionDetailResponse:
    """Add a game to a collection."""
    return await service.add_game(collection_id, current_user, data)


@router.delete(
    "/{collection_id}/games/{game_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a game from a collection",
)
async def remove_game_from_collection(
    collection_id: UUID,
    game_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[CollectionService, Depends(get_collection_service)],
) -> None:
    """Remove a game from a collection."""
    await service.remove_game(collection_id, game_id, current_user)


@router.put(
    "/{collection_id}/games/reorder",
    response_model=CollectionDetailResponse,
    summary="Reorder games in a collection",
)
async def reorder_collection_games(
    collection_id: UUID,
    data: ReorderRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[CollectionService, Depends(get_collection_service)],
) -> CollectionDetailResponse:
    """Reorder games within a collection."""
    return await service.reorder_games(collection_id, current_user, data.game_ids)
