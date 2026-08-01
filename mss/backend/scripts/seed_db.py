"""Seed the development database with sample data."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.database import async_session_factory, engine
from app.models.base import Base, UserRole
from app.models.forum import ForumCategory
from app.models.game import Engine, Game, Tag, TagCategory
from app.models.review import Review
from app.models.user import User, UserProfile
from app.utils.security import hash_password
from app.utils.slugify import generate_slug


async def seed() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_factory() as session:
        existing = await session.execute(select(User).limit(1))
        if existing.scalar_one_or_none():
            print("Database already seeded, skipping.")
            return

        admin = User(
            username="admin",
            email="admin@gamevault.dev",
            password_hash=hash_password("adminpass123"),
            display_name="Admin",
            is_verified=True,
            role=UserRole.ADMIN,
        )
        demo_user = User(
            username="demouser",
            email="demo@gamevault.dev",
            password_hash=hash_password("demopass123"),
            display_name="Demo User",
            is_verified=True,
        )
        session.add_all([admin, demo_user])
        await session.flush()

        session.add_all([
            UserProfile(user_id=admin.id),
            UserProfile(user_id=demo_user.id),
        ])

        engines = [
            Engine(name="RPG Maker MV", slug="rpg-maker-mv"),
            Engine(name="Twine", slug="twine"),
            Engine(name="Unity", slug="unity"),
        ]
        session.add_all(engines)
        await session.flush()

        tags = [
            Tag(name="Fantasy", slug="fantasy", category=TagCategory.GENRE),
            Tag(name="Romance", slug="romance", category=TagCategory.GENRE),
            Tag(name="Adventure", slug="adventure", category=TagCategory.GENRE),
        ]
        session.add_all(tags)
        await session.flush()

        games = [
            Game(
                title="Sample Adventure",
                slug=generate_slug("Sample Adventure"),
                synopsis="A sample game for development.",
                engine_id=engines[0].id,
                author_id=demo_user.id,
                is_featured=True,
            ),
            Game(
                title="Twine Story Demo",
                slug=generate_slug("Twine Story Demo"),
                synopsis="An interactive fiction demo.",
                engine_id=engines[1].id,
                author_id=demo_user.id,
            ),
        ]
        session.add_all(games)
        await session.flush()

        for game in games:
            game.tags.append(tags[0])

        reviews = [
            Review(
                game_id=games[0].id,
                user_id=admin.id,
                score=8,
                body="Great sample game for testing!",
            ),
        ]
        session.add_all(reviews)

        categories = [
            ForumCategory(
                name="General Discussion",
                slug="general",
                description="Talk about anything game-related.",
                sort_order=1,
            ),
            ForumCategory(
                name="Game Development",
                slug="game-development",
                description="Share tips and resources for game creation.",
                sort_order=2,
            ),
        ]
        session.add_all(categories)

        await session.commit()
        print("Database seeded successfully.")


if __name__ == "__main__":
    asyncio.run(seed())
