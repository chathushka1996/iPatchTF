from functools import lru_cache

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "GameVault"
    APP_ENV: str = "development"
    SECRET_KEY: str = "change-me-to-a-64-char-random-hex"
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost"

    # Database
    DB_HOST: str = "postgres"
    DB_PORT: int = 5432
    DB_NAME: str = "gamevault"
    DB_USER: str = "gamevault"
    DB_PASSWORD: str = "supersecretpassword"
    DATABASE_URL: str | None = None

    # Redis / Celery
    REDIS_URL: str = "redis://redis:6379/0"
    CELERY_BROKER_URL: str = "redis://redis:6379/1"

    # Meilisearch
    MEILI_URL: str = "http://meilisearch:7700"
    MEILI_MASTER_KEY: str = "masterkey-change-in-production"

    # MinIO
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_GAMES: str = "game-files"
    MINIO_BUCKET_SCREENSHOTS: str = "screenshots"
    MINIO_BUCKET_AVATARS: str = "avatars"

    # Auth
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Email
    SMTP_HOST: str = "smtp.example.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = "noreply@gamevault.dev"
    SMTP_PASSWORD: str = ""
    EMAIL_FROM: str = "GameVault <noreply@gamevault.dev>"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
