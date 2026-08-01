import uuid
from datetime import timedelta
from typing import Literal

from minio import Minio
from minio.error import S3Error

from app.config import settings
from app.exceptions import ValidationError

UploadPurpose = Literal["avatar", "screenshot", "game_file", "forum_attachment"]

UPLOAD_LIMITS: dict[UploadPurpose, dict[str, int | list[str]]] = {
    "avatar": {
        "max_size": 5 * 1024 * 1024,
        "content_types": ["image/jpeg", "image/png", "image/webp", "image/gif"],
    },
    "screenshot": {
        "max_size": 10 * 1024 * 1024,
        "content_types": ["image/jpeg", "image/png", "image/webp"],
    },
    "game_file": {
        "max_size": 2 * 1024 * 1024 * 1024,
        "content_types": [
            "application/zip",
            "application/x-zip-compressed",
            "application/octet-stream",
        ],
    },
    "forum_attachment": {
        "max_size": 20 * 1024 * 1024,
        "content_types": [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/gif",
            "application/pdf",
            "application/zip",
        ],
    },
}

BUCKET_MAP: dict[UploadPurpose, str] = {
    "avatar": settings.MINIO_BUCKET_AVATARS,
    "screenshot": settings.MINIO_BUCKET_SCREENSHOTS,
    "game_file": settings.MINIO_BUCKET_GAMES,
    "forum_attachment": settings.MINIO_BUCKET_SCREENSHOTS,
}


class UploadService:
    def __init__(self) -> None:
        self._client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_ENDPOINT.startswith("https"),
        )

    def validate_upload(
        self,
        purpose: UploadPurpose,
        content_type: str,
        size: int,
    ) -> None:
        limits = UPLOAD_LIMITS.get(purpose)
        if not limits:
            raise ValidationError(detail=f"Unknown upload purpose: {purpose}")

        max_size = limits["max_size"]
        if size > max_size:
            raise ValidationError(
                detail=f"File exceeds maximum size of {max_size} bytes for {purpose}"
            )

        allowed_types: list[str] = limits["content_types"]  # type: ignore[assignment]
        if content_type not in allowed_types:
            raise ValidationError(
                detail=f"Content type '{content_type}' not allowed for {purpose}"
            )

    def generate_presigned_url(
        self,
        filename: str,
        content_type: str,
        purpose: UploadPurpose,
    ) -> tuple[str, str]:
        self.validate_upload(purpose, content_type, size=1)

        bucket = BUCKET_MAP[purpose]
        ext = filename.rsplit(".", 1)[-1] if "." in filename else "bin"
        object_key = f"{purpose}/{uuid.uuid4()}.{ext}"

        upload_url = self._client.presigned_put_object(
            bucket,
            object_key,
            expires=timedelta(hours=1),
        )
        return upload_url, object_key

    def verify_object_exists(self, object_key: str) -> bool:
        bucket = object_key.split("/")[0] if "/" not in object_key else None
        for candidate_bucket in BUCKET_MAP.values():
            try:
                self._client.stat_object(candidate_bucket, object_key)
                return True
            except S3Error:
                continue
        return False

    def delete_object(self, object_key: str) -> None:
        for bucket in BUCKET_MAP.values():
            try:
                self._client.remove_object(bucket, object_key)
                return
            except S3Error:
                continue
