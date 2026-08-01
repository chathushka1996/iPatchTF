from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=24, ge=1, le=100)


class PaginatedResponse(BaseModel, Generic[T]):
    model_config = ConfigDict(from_attributes=True)

    items: list[T]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    per_page: int = Field(ge=1, le=100)
    pages: int = Field(ge=0)


class ErrorResponse(BaseModel):
    detail: str
    error_code: str
    status: int


class HealthResponse(BaseModel):
    status: str
    database: str
    redis: str
    meilisearch: str
    minio: str


class MessageResponse(BaseModel):
    message: str
