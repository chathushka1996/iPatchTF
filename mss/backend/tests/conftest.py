import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import get_async_session
from app.models import collection, forum, game, notification, report, review, user  # noqa: F401
from app.models.base import Base
from app.models.game import Game
from app.models.review import Review
from app.models.user import User, UserProfile
from app.utils.security import hash_password

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(TEST_DATABASE_URL, echo=False)
test_session_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with test_session_factory() as session:
        yield session
        await session.rollback()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    try:
        from app.main import app
    except ImportError:
        pytest.skip("app.main not available")

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_async_session] = override_get_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


async def create_test_user(
    session: AsyncSession,
    username: str | None = None,
    email: str | None = None,
    **kwargs: Any,
) -> User:
    user = User(
        username=username or f"user_{uuid4().hex[:8]}",
        email=email or f"{uuid4().hex[:8]}@test.com",
        password_hash=hash_password("testpass123"),
        display_name="Test User",
        is_verified=True,
        **kwargs,
    )
    session.add(user)
    await session.flush()
    session.add(UserProfile(user_id=user.id))
    await session.flush()
    return user


async def create_test_game(
    session: AsyncSession,
    author: User,
    title: str = "Test Game",
    **kwargs: Any,
) -> Game:
    game = Game(
        title=title,
        slug=kwargs.pop("slug", f"test-game-{uuid4().hex[:8]}"),
        author_id=author.id,
        synopsis="A test game",
        **kwargs,
    )
    session.add(game)
    await session.flush()
    return game


async def create_test_review(
    session: AsyncSession,
    game: Game,
    user: User,
    score: int = 8,
    body: str = "Great game!",
) -> Review:
    review = Review(
        game_id=game.id,
        user_id=user.id,
        score=score,
        body=body,
    )
    session.add(review)
    await session.flush()
    return review


@pytest.fixture
def user_factory(db_session: AsyncSession):
    async def _factory(**kwargs: Any) -> User:
        return await create_test_user(db_session, **kwargs)

    return _factory


@pytest.fixture
def game_factory(db_session: AsyncSession):
    async def _factory(author: User, **kwargs: Any) -> Game:
        return await create_test_game(db_session, author, **kwargs)

    return _factory


@pytest.fixture
def review_factory(db_session: AsyncSession):
    async def _factory(game: Game, user: User, **kwargs: Any) -> Review:
        return await create_test_review(db_session, game, user, **kwargs)

    return _factory
