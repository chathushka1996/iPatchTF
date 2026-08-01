from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.schemas.forum import ChatMessageResponse
from app.services.forum_service import ForumService

router = APIRouter(prefix="/chat", tags=["chat"])
ws_router = APIRouter(tags=["websocket"])


class ChatChannelResponse(BaseModel):
    name: str
    label: str
    description: str | None = None


def get_forum_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ForumService:
    return ForumService(db)


@router.get(
    "/channels",
    response_model=list[ChatChannelResponse],
    summary="List chat channels",
)
async def list_chat_channels(
    service: Annotated[ForumService, Depends(get_forum_service)],
) -> list[ChatChannelResponse]:
    """Return available chat channels."""
    return await service.list_chat_channels()


@router.get(
    "/{channel}/messages",
    response_model=list[ChatMessageResponse],
    summary="Get chat message history",
)
async def get_chat_messages(
    channel: str,
    service: Annotated[ForumService, Depends(get_forum_service)],
) -> list[ChatMessageResponse]:
    """Return the last 50 messages for a chat channel."""
    return await service.get_chat_messages(channel, limit=50)


@ws_router.websocket("/chat/{channel}")
async def chat_websocket(
    websocket: WebSocket,
    channel: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """WebSocket endpoint for real-time chat in a channel."""
    service = ForumService(db)
    await websocket.accept()
    try:
        await service.handle_chat_websocket(websocket, channel)
    except WebSocketDisconnect:
        await service.disconnect_chat_websocket(websocket, channel)


@ws_router.websocket("/notifications")
async def notifications_websocket(
    websocket: WebSocket,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """WebSocket endpoint for real-time user notifications."""
    service = ForumService(db)
    await websocket.accept()
    try:
        await service.handle_notifications_websocket(websocket)
    except WebSocketDisconnect:
        await service.disconnect_notifications_websocket(websocket)
