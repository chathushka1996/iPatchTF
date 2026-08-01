import enum
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class OriginalPCGender(str, enum.Enum):
    MALE = "male"
    FEMALE = "female"
    SELECTABLE = "selectable"
    GENDERLESS = "genderless"
    HERMAPHRODITE = "hermaphrodite"


class ContentRating(str, enum.Enum):
    G = "G"
    PG = "PG"
    R = "R"
    X = "X"
    XXX = "XXX"


class DevelopmentStatus(str, enum.Enum):
    CONCEPT = "concept"
    DEMO = "demo"
    ALPHA = "alpha"
    BETA = "beta"
    COMPLETE = "complete"
    DISCONTINUED = "discontinued"


class TagCategory(str, enum.Enum):
    GENRE = "genre"
    ADULT_THEME = "adult_theme"
    TRANSFORMATION = "transformation"
    MULTIMEDIA = "multimedia"
    CONTENT_WARNING = "content_warning"
    PLATFORM = "platform"


class Engine(Base):
    __tablename__ = "engines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    game_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    games: Mapped[list["Game"]] = relationship(back_populates="engine")


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    category: Mapped[TagCategory] = mapped_column(
        Enum(TagCategory, name="tag_category", native_enum=False),
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(Text)

    games: Mapped[list["Game"]] = relationship(
        secondary="game_tags",
        back_populates="tags",
    )


class Game(Base, TimestampMixin):
    __tablename__ = "games"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    slug: Mapped[str] = mapped_column(String(300), unique=True, nullable=False, index=True)
    synopsis: Mapped[str | None] = mapped_column(Text)
    plot: Mapped[str | None] = mapped_column(Text)
    characters: Mapped[str | None] = mapped_column(Text)
    walkthrough: Mapped[str | None] = mapped_column(Text)
    engine_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("engines.id", ondelete="SET NULL"),
    )
    author_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    original_pc_gender: Mapped[OriginalPCGender | None] = mapped_column(
        Enum(OriginalPCGender, name="original_pc_gender", native_enum=False),
    )
    rating: Mapped[ContentRating | None] = mapped_column(
        Enum(ContentRating, name="content_rating", native_enum=False),
    )
    development_status: Mapped[DevelopmentStatus | None] = mapped_column(
        Enum(DevelopmentStatus, name="development_status", native_enum=False),
    )
    is_free: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    has_purchasable_content: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    support_url: Mapped[str | None] = mapped_column(String(500))
    language: Mapped[str] = mapped_column(String(50), default="English", nullable=False)
    play_online_url: Mapped[str | None] = mapped_column(String(500))
    like_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    review_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    average_score: Mapped[Decimal] = mapped_column(Numeric(3, 2), default=Decimal("0.00"), nullable=False)
    view_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    play_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_featured: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_approved: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    search_vector: Mapped[str | None] = mapped_column(TSVECTOR)

    engine: Mapped["Engine | None"] = relationship(back_populates="games")
    versions: Mapped[list["GameVersion"]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
    )
    screenshots: Mapped[list["GameScreenshot"]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
    )
    tags: Mapped[list["Tag"]] = relationship(
        secondary="game_tags",
        back_populates="games",
    )
    authors: Mapped[list["GameAuthor"]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
    )
    likes: Mapped[list["GameLike"]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
    )
    follows: Mapped[list["GameFollow"]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
    )


class GameVersion(Base):
    __tablename__ = "game_versions"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    game_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("games.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_string: Mapped[str] = mapped_column(String(50), nullable=False)
    changelog: Mapped[str | None] = mapped_column(Text)
    release_date: Mapped[date | None] = mapped_column(Date)
    is_latest: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    game: Mapped["Game"] = relationship(back_populates="versions")
    downloads: Mapped[list["GameVersionDownload"]] = relationship(
        back_populates="version",
        cascade="all, delete-orphan",
    )


class GameVersionDownload(Base):
    __tablename__ = "game_version_downloads"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    version_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("game_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str | None] = mapped_column(String(100))
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    platform: Mapped[str | None] = mapped_column(String(50))

    version: Mapped["GameVersion"] = relationship(back_populates="downloads")


class GameScreenshot(Base):
    __tablename__ = "game_screenshots"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    game_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("games.id", ondelete="CASCADE"),
        nullable=False,
    )
    image_url: Mapped[str] = mapped_column(Text, nullable=False)
    thumbnail_url: Mapped[str | None] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(String(255))
    sort_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    game: Mapped["Game"] = relationship(back_populates="screenshots")


class GameTag(Base):
    __tablename__ = "game_tags"

    game_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("games.id", ondelete="CASCADE"),
        primary_key=True,
    )
    tag_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tags.id", ondelete="CASCADE"),
        primary_key=True,
    )


class GameAuthor(Base):
    __tablename__ = "game_authors"

    game_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("games.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    role: Mapped[str] = mapped_column(String(50), nullable=False)

    game: Mapped["Game"] = relationship(back_populates="authors")


class GameLike(Base):
    __tablename__ = "game_likes"

    game_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("games.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    game: Mapped["Game"] = relationship(back_populates="likes")


class GameFollow(Base):
    __tablename__ = "game_follows"

    game_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("games.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    game: Mapped["Game"] = relationship(back_populates="follows")
