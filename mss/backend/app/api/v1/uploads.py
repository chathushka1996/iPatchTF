from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_active_user, get_db
from app.models.user import User
from app.schemas.upload import PresignRequest, PresignResponse
from app.services.upload_service import UploadService

router = APIRouter(prefix="/uploads", tags=["uploads"])


def get_upload_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UploadService:
    return UploadService(db)


@router.post(
    "/presign",
    response_model=PresignResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a presigned upload URL",
)
async def generate_presigned_url(
    data: PresignRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    service: Annotated[UploadService, Depends(get_upload_service)],
) -> PresignResponse:
    """Generate a presigned URL for direct upload to object storage."""
    return await service.generate_presigned_url(current_user, data)
