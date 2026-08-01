from typing import Any, Generic, TypeVar
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

ModelT = TypeVar("ModelT")


class BaseRepository(Generic[ModelT]):
    def __init__(self, model: type[ModelT], session: AsyncSession) -> None:
        self.model = model
        self.session = session

    async def get_by_id(self, id: UUID | int) -> ModelT | None:
        return await self.session.get(self.model, id)

    async def get_all(self, skip: int = 0, limit: int = 100) -> list[ModelT]:
        result = await self.session.execute(
            select(self.model).offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def create(self, data: dict[str, Any]) -> ModelT:
        instance = self.model(**data)
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def update(self, id: UUID | int, data: dict[str, Any]) -> ModelT | None:
        await self.session.execute(
            update(self.model).where(self.model.id == id).values(**data)
        )
        await self.session.flush()
        return await self.get_by_id(id)

    async def delete(self, id: UUID | int) -> bool:
        result = await self.session.execute(
            delete(self.model).where(self.model.id == id)
        )
        await self.session.flush()
        return result.rowcount > 0

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        query = select(func.count()).select_from(self.model)
        if filters:
            for key, value in filters.items():
                query = query.where(getattr(self.model, key) == value)
        result = await self.session.execute(query)
        return result.scalar_one()
