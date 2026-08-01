# GameVault — Interactive Game Database & Community Platform

## Full-Stack Requirements Document

**Version:** 1.0.0
**Date:** July 31, 2026
**Status:** Draft

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Directory Structure](#3-directory-structure)
4. [Docker Compose Infrastructure](#4-docker-compose-infrastructure)
5. [Database Schema Design](#5-database-schema-design)
6. [Backend Requirements (FastAPI)](#6-backend-requirements-fastapi)
7. [Frontend Requirements (Next.js)](#7-frontend-requirements-nextjs)
8. [Feature Specifications](#8-feature-specifications)
9. [Non-Functional Requirements](#9-non-functional-requirements)
10. [Deployment & DevOps](#10-deployment--devops)
11. [Appendix: Suggested Enhancements Beyond Reference](#11-appendix-suggested-enhancements-beyond-reference)

---

## 1. Executive Summary

GameVault is a community-driven interactive game database and social platform inspired by tfgames.site's IGDB. It allows users to discover, submit, review, and discuss games across a wide range of engines, genres, and themes. The platform combines a rich game catalog with a full community layer (forums, real-time chat, discussion threads) and a modern, polished UI.

### 1.1 Core Goals

- Provide a searchable, filterable game database with rich metadata (engine, genre, themes, multimedia type, development status, ratings).
- Enable registered users to submit new game entries, update existing ones with new versions, write reviews, and curate personal collections.
- Foster community engagement through per-game discussion threads, general forums, real-time chat, and a notification system.
- Ship as a fully containerized stack (Docker Compose) that any developer can spin up in a single command.

### 1.2 Tech Stack Summary

| Layer | Technology | Justification |
|---|---|---|
| Frontend | **Next.js 14+ (App Router)** | SSR for SEO on game pages, React Server Components for performance, file-based routing. |
| Backend API | **FastAPI (Python 3.12+)** | Async-first, auto-generated OpenAPI docs, Pydantic validation, WebSocket support built-in. |
| Database | **PostgreSQL 16** | Robust relational storage, full-text search via `tsvector`, JSONB for flexible metadata. |
| Cache | **Redis 7** | Session store, rate limiting, pub/sub for real-time features, background task broker. |
| Search Engine | **Meilisearch** | Typo-tolerant instant search, faceted filtering, lightweight and fast to deploy. |
| Object Storage | **MinIO** | S3-compatible self-hosted storage for game files, screenshots, avatars. |
| Background Tasks | **Celery** (with Redis broker) | Email sending, search index sync, file processing, notification dispatch. |
| Reverse Proxy | **Traefik** | Auto-discovery of Docker services, built-in TLS, rate limiting middleware. |
| Real-time | **FastAPI WebSockets** + Redis pub/sub | Live chat, real-time notifications, presence indicators. |

---

## 2. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         TRAEFIK (Reverse Proxy)                  │
│                    :80 / :443  ─ TLS termination                 │
└─────────┬──────────────────────────────────┬─────────────────────┘
          │                                  │
          │  /api/*  /ws/*                   │  /*  (everything else)
          ▼                                  ▼
┌──────────────────────┐          ┌──────────────────────────┐
│   FASTAPI BACKEND    │          │   NEXT.JS FRONTEND       │
│   (uvicorn, 4 workers)│         │   (node, SSR + static)   │
│   Port: 8000         │          │   Port: 3000             │
└──────┬───────┬───────┘          └──────────────────────────┘
       │       │
       │       │  pub/sub + cache + broker
       │       ▼
       │  ┌──────────┐     ┌──────────────┐
       │  │  REDIS    │     │  CELERY       │
       │  │  :6379    │◄────│  WORKER       │
       │  └──────────┘     └──────┬───────┘
       │                          │
       ▼                          ▼
┌──────────────┐         ┌──────────────┐    ┌──────────────┐
│ POSTGRESQL   │         │ MEILISEARCH  │    │    MINIO     │
│ :5432        │         │ :7700        │    │ :9000/:9001  │
└──────────────┘         └──────────────┘    └──────────────┘
```

### 2.1 Service Communication Flow

- **Client → Traefik**: All traffic enters through Traefik. Routes matching `/api/*` and `/ws/*` are forwarded to FastAPI; everything else goes to Next.js.
- **FastAPI → PostgreSQL**: Primary data persistence. SQLAlchemy async ORM with Alembic migrations.
- **FastAPI → Redis**: JWT blocklist, session data, rate limiting counters, pub/sub channels for WebSocket fan-out, Celery task broker.
- **FastAPI → Meilisearch**: On every game create/update/delete, a Celery task synchronizes the record to the Meilisearch index. The frontend hits FastAPI's search endpoint, which proxies to Meilisearch.
- **FastAPI → MinIO**: Presigned URL generation for direct browser uploads (screenshots, game files, avatars). FastAPI validates and records the metadata; the actual bytes never pass through the API server.
- **Celery Worker**: Consumes tasks from Redis. Handles email dispatch (via SMTP), search index sync, image thumbnail generation, periodic cleanup jobs, and notification fan-out.
- **Next.js → FastAPI**: Server-side fetches during SSR (for SEO-critical pages like game detail) and client-side fetches via the API layer.

---

## 3. Directory Structure

```
gamevault/
│
├── docker-compose.yml              # Orchestration for all services
├── docker-compose.override.yml     # Dev-only overrides (volumes, hot reload)
├── .env.example                    # Template environment variables
├── .env                            # Local environment (gitignored)
├── Makefile                        # Common commands (make up, make migrate, etc.)
├── README.md                       # Project setup and contribution guide
│
├── backend/
│   ├── Dockerfile                  # Multi-stage: builder + slim runtime
│   ├── pyproject.toml              # Dependencies (Poetry or uv)
│   ├── alembic/                    # Database migration scripts
│   │   ├── env.py
│   │   ├── alembic.ini
│   │   └── versions/              # Auto-generated migration files
│   ├── app/
│   │   ├── main.py                 # FastAPI application factory
│   │   ├── config.py               # Pydantic Settings (env-based config)
│   │   ├── dependencies.py         # Shared DI (get_db, get_current_user, etc.)
│   │   ├── exceptions.py           # Custom exception classes + handlers
│   │   ├── middleware/
│   │   │   ├── cors.py
│   │   │   ├── rate_limit.py
│   │   │   └── request_id.py
│   │   ├── models/                 # SQLAlchemy ORM models
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   ├── game.py
│   │   │   ├── review.py
│   │   │   ├── forum.py
│   │   │   ├── notification.py
│   │   │   ├── tag.py
│   │   │   └── collection.py
│   │   ├── schemas/                # Pydantic request/response schemas
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   ├── game.py
│   │   │   ├── review.py
│   │   │   ├── forum.py
│   │   │   ├── notification.py
│   │   │   └── common.py           # Pagination, error, health schemas
│   │   ├── api/                    # Route handlers grouped by domain
│   │   │   ├── __init__.py
│   │   │   ├── v1/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── router.py       # Aggregates all v1 routes
│   │   │   │   ├── auth.py
│   │   │   │   ├── users.py
│   │   │   │   ├── games.py
│   │   │   │   ├── reviews.py
│   │   │   │   ├── forums.py
│   │   │   │   ├── threads.py
│   │   │   │   ├── chat.py
│   │   │   │   ├── notifications.py
│   │   │   │   ├── collections.py
│   │   │   │   ├── search.py
│   │   │   │   ├── admin.py
│   │   │   │   └── uploads.py
│   │   │   └── websocket.py        # WebSocket connection manager
│   │   ├── services/               # Business logic layer
│   │   │   ├── auth_service.py
│   │   │   ├── game_service.py
│   │   │   ├── review_service.py
│   │   │   ├── forum_service.py
│   │   │   ├── search_service.py
│   │   │   ├── notification_service.py
│   │   │   ├── upload_service.py
│   │   │   ├── moderation_service.py
│   │   │   └── email_service.py
│   │   ├── repositories/           # Data access layer (queries)
│   │   │   ├── base.py
│   │   │   ├── user_repo.py
│   │   │   ├── game_repo.py
│   │   │   ├── review_repo.py
│   │   │   └── forum_repo.py
│   │   ├── tasks/                  # Celery task definitions
│   │   │   ├── __init__.py         # Celery app initialization
│   │   │   ├── email_tasks.py
│   │   │   ├── search_tasks.py
│   │   │   ├── notification_tasks.py
│   │   │   └── cleanup_tasks.py
│   │   └── utils/
│   │       ├── security.py         # Password hashing, JWT encode/decode
│   │       ├── pagination.py       # Cursor and offset pagination helpers
│   │       ├── slugify.py          # URL-safe slug generation
│   │       └── validators.py       # Shared field validators
│   ├── tests/
│   │   ├── conftest.py             # Fixtures (test DB, test client, factory boy)
│   │   ├── unit/
│   │   ├── integration/
│   │   └── e2e/
│   └── scripts/
│       ├── seed_db.py              # Populate dev database with sample data
│       └── create_superuser.py     # CLI to create first admin account
│
├── frontend/
│   ├── Dockerfile                  # Multi-stage: deps install + build + serve
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   ├── next.config.mjs
│   ├── postcss.config.mjs
│   ├── public/
│   │   ├── favicon.ico
│   │   └── images/                 # Static brand assets
│   ├── src/
│   │   ├── app/                    # Next.js App Router pages
│   │   │   ├── layout.tsx          # Root layout (providers, nav, footer)
│   │   │   ├── page.tsx            # Homepage
│   │   │   ├── (auth)/
│   │   │   │   ├── login/page.tsx
│   │   │   │   ├── register/page.tsx
│   │   │   │   ├── forgot-password/page.tsx
│   │   │   │   └── reset-password/page.tsx
│   │   │   ├── games/
│   │   │   │   ├── page.tsx        # Browse / search results
│   │   │   │   ├── [slug]/
│   │   │   │   │   ├── page.tsx    # Game detail
│   │   │   │   │   └── discussion/page.tsx
│   │   │   │   ├── submit/page.tsx
│   │   │   │   └── [slug]/edit/page.tsx
│   │   │   ├── browse/
│   │   │   │   └── [category]/page.tsx  # By engine, genre, author, etc.
│   │   │   ├── community/
│   │   │   │   ├── page.tsx        # Forum index
│   │   │   │   ├── [forumSlug]/page.tsx
│   │   │   │   └── [forumSlug]/[threadSlug]/page.tsx
│   │   │   ├── chat/page.tsx       # Real-time chat
│   │   │   ├── profile/
│   │   │   │   ├── [username]/page.tsx    # Public profile
│   │   │   │   └── settings/page.tsx      # Account settings
│   │   │   ├── collections/
│   │   │   │   ├── page.tsx
│   │   │   │   └── [id]/page.tsx
│   │   │   ├── dashboard/
│   │   │   │   ├── page.tsx        # User dashboard
│   │   │   │   ├── my-games/page.tsx
│   │   │   │   ├── my-reviews/page.tsx
│   │   │   │   └── notifications/page.tsx
│   │   │   ├── admin/
│   │   │   │   ├── layout.tsx
│   │   │   │   ├── page.tsx        # Admin dashboard
│   │   │   │   ├── users/page.tsx
│   │   │   │   ├── games/page.tsx
│   │   │   │   ├── reports/page.tsx
│   │   │   │   └── analytics/page.tsx
│   │   │   └── not-found.tsx
│   │   ├── components/
│   │   │   ├── ui/                 # Atomic UI components (shadcn/ui)
│   │   │   │   ├── button.tsx
│   │   │   │   ├── input.tsx
│   │   │   │   ├── dialog.tsx
│   │   │   │   ├── dropdown-menu.tsx
│   │   │   │   ├── badge.tsx
│   │   │   │   ├── card.tsx
│   │   │   │   ├── tabs.tsx
│   │   │   │   ├── toast.tsx
│   │   │   │   ├── skeleton.tsx
│   │   │   │   └── ...
│   │   │   ├── layout/
│   │   │   │   ├── navbar.tsx
│   │   │   │   ├── sidebar.tsx
│   │   │   │   ├── footer.tsx
│   │   │   │   ├── mobile-nav.tsx
│   │   │   │   └── theme-toggle.tsx
│   │   │   ├── games/
│   │   │   │   ├── game-card.tsx
│   │   │   │   ├── game-grid.tsx
│   │   │   │   ├── game-detail-header.tsx
│   │   │   │   ├── game-metadata-sidebar.tsx
│   │   │   │   ├── game-version-history.tsx
│   │   │   │   ├── game-submit-form.tsx
│   │   │   │   ├── game-screenshot-gallery.tsx
│   │   │   │   └── similar-games.tsx
│   │   │   ├── search/
│   │   │   │   ├── search-bar.tsx
│   │   │   │   ├── filter-panel.tsx
│   │   │   │   ├── active-filters.tsx
│   │   │   │   └── sort-selector.tsx
│   │   │   ├── reviews/
│   │   │   │   ├── review-card.tsx
│   │   │   │   ├── review-form.tsx
│   │   │   │   ├── review-list.tsx
│   │   │   │   └── star-rating.tsx
│   │   │   ├── community/
│   │   │   │   ├── forum-category-card.tsx
│   │   │   │   ├── thread-list.tsx
│   │   │   │   ├── thread-post.tsx
│   │   │   │   ├── post-editor.tsx
│   │   │   │   └── chat-window.tsx
│   │   │   ├── profile/
│   │   │   │   ├── avatar-upload.tsx
│   │   │   │   ├── profile-header.tsx
│   │   │   │   ├── activity-feed.tsx
│   │   │   │   └── user-stats.tsx
│   │   │   └── shared/
│   │   │       ├── pagination.tsx
│   │   │       ├── empty-state.tsx
│   │   │       ├── error-boundary.tsx
│   │   │       ├── loading-spinner.tsx
│   │   │       ├── confirm-dialog.tsx
│   │   │       ├── markdown-renderer.tsx
│   │   │       ├── notification-bell.tsx
│   │   │       └── report-button.tsx
│   │   ├── hooks/
│   │   │   ├── use-auth.ts
│   │   │   ├── use-debounce.ts
│   │   │   ├── use-infinite-scroll.ts
│   │   │   ├── use-websocket.ts
│   │   │   ├── use-search.ts
│   │   │   └── use-media-query.ts
│   │   ├── lib/
│   │   │   ├── api-client.ts       # Axios/fetch wrapper with interceptors
│   │   │   ├── auth.ts             # Token management, refresh logic
│   │   │   ├── utils.ts            # cn(), formatDate(), etc.
│   │   │   ├── constants.ts
│   │   │   └── validators.ts       # Zod schemas for form validation
│   │   ├── providers/
│   │   │   ├── auth-provider.tsx
│   │   │   ├── theme-provider.tsx
│   │   │   ├── toast-provider.tsx
│   │   │   └── query-provider.tsx   # TanStack Query
│   │   ├── stores/                  # Zustand stores
│   │   │   ├── auth-store.ts
│   │   │   ├── notification-store.ts
│   │   │   └── chat-store.ts
│   │   ├── types/
│   │   │   ├── api.ts              # Response types mirroring backend schemas
│   │   │   ├── game.ts
│   │   │   ├── user.ts
│   │   │   └── forum.ts
│   │   └── styles/
│   │       └── globals.css          # Tailwind directives + CSS custom properties
│   └── tests/
│       ├── components/
│       └── e2e/                     # Playwright tests
│
├── nginx/                           # (Alternative to Traefik, simpler setups)
│   └── default.conf
│
├── scripts/
│   ├── init-minio.sh                # Create default buckets on first run
│   ├── init-db.sh                   # Run migrations + seed
│   └── backup-db.sh                 # pg_dump scheduled backup
│
└── docs/
    ├── api.md                       # API endpoint reference
    ├── deployment.md                # Production deployment guide
    ├── contributing.md              # Contributor guidelines
    └── architecture.md              # ADRs and design decisions
```

---

## 4. Docker Compose Infrastructure

### 4.1 Services Definition

```yaml
# docker-compose.yml (conceptual — not runnable as-is, but the blueprint)

version: "3.9"

services:

  # ─── REVERSE PROXY ──────────────────────────────────────────
  traefik:
    image: traefik:v3.1
    ports:
      - "80:80"
      - "443:443"
      - "8080:8080"        # Traefik dashboard (dev only)
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./traefik/traefik.yml:/etc/traefik/traefik.yml:ro
    depends_on:
      - backend
      - frontend

  # ─── BACKEND API ─────────────────────────────────────────────
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
      target: development       # multi-stage: dev has hot-reload
    env_file: .env
    volumes:
      - ./backend/app:/app/app  # Hot-reload in dev
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      meilisearch:
        condition: service_started
      minio:
        condition: service_started
    labels:
      - "traefik.http.routers.api.rule=PathPrefix(`/api`) || PathPrefix(`/ws`)"
      - "traefik.http.services.api.loadbalancer.server.port=8000"

  # ─── CELERY WORKER ──────────────────────────────────────────
  celery-worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
      target: development
    command: celery -A app.tasks worker --loglevel=info --concurrency=4
    env_file: .env
    depends_on:
      - redis
      - postgres

  # ─── CELERY BEAT (scheduler) ─────────────────────────────────
  celery-beat:
    build:
      context: ./backend
      dockerfile: Dockerfile
      target: development
    command: celery -A app.tasks beat --loglevel=info
    env_file: .env
    depends_on:
      - redis

  # ─── FRONTEND ────────────────────────────────────────────────
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      target: development
    env_file: .env
    volumes:
      - ./frontend/src:/app/src  # Hot-reload in dev
    labels:
      - "traefik.http.routers.web.rule=PathPrefix(`/`)"
      - "traefik.http.services.web.loadbalancer.server.port=3000"

  # ─── DATABASE ────────────────────────────────────────────────
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: ${DB_NAME}
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 5s
      retries: 5
    ports:
      - "5432:5432"             # Exposed for dev tooling

  # ─── CACHE / BROKER ─────────────────────────────────────────
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      retries: 5

  # ─── SEARCH ENGINE ──────────────────────────────────────────
  meilisearch:
    image: getmeili/meilisearch:v1.9
    environment:
      MEILI_MASTER_KEY: ${MEILI_MASTER_KEY}
      MEILI_ENV: development
    volumes:
      - meili_data:/meili_data

  # ─── OBJECT STORAGE ─────────────────────────────────────────
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ACCESS_KEY}
      MINIO_ROOT_PASSWORD: ${MINIO_SECRET_KEY}
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"             # MinIO console

volumes:
  postgres_data:
  redis_data:
  meili_data:
  minio_data:
```

### 4.2 Environment Variables (.env.example)

```env
# ─── APPLICATION ──────────────────────────────
APP_NAME=GameVault
APP_ENV=development
SECRET_KEY=change-me-to-a-64-char-random-hex
ALLOWED_ORIGINS=http://localhost:3000,http://localhost

# ─── DATABASE ─────────────────────────────────
DB_HOST=postgres
DB_PORT=5432
DB_NAME=gamevault
DB_USER=gamevault
DB_PASSWORD=supersecretpassword
DATABASE_URL=postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}

# ─── REDIS ────────────────────────────────────
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/1

# ─── MEILISEARCH ──────────────────────────────
MEILI_URL=http://meilisearch:7700
MEILI_MASTER_KEY=masterkey-change-in-production

# ─── MINIO ────────────────────────────────────
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_GAMES=game-files
MINIO_BUCKET_SCREENSHOTS=screenshots
MINIO_BUCKET_AVATARS=avatars

# ─── AUTH ─────────────────────────────────────
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# ─── EMAIL ────────────────────────────────────
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=noreply@gamevault.dev
SMTP_PASSWORD=emailpassword
EMAIL_FROM=GameVault <noreply@gamevault.dev>

# ─── FRONTEND ────────────────────────────────
NEXT_PUBLIC_API_URL=http://localhost/api
NEXT_PUBLIC_WS_URL=ws://localhost/ws
NEXT_PUBLIC_MINIO_PUBLIC_URL=http://localhost:9000
```

### 4.3 Makefile Commands

```makefile
up:              docker compose up -d --build
down:            docker compose down
logs:            docker compose logs -f
migrate:         docker compose exec backend alembic upgrade head
makemigrations:  docker compose exec backend alembic revision --autogenerate -m "$(msg)"
seed:            docker compose exec backend python scripts/seed_db.py
superuser:       docker compose exec backend python scripts/create_superuser.py
test-backend:    docker compose exec backend pytest -v
test-frontend:   docker compose exec frontend npm test
shell:           docker compose exec backend python -i
reset-db:        docker compose down -v && make up && sleep 5 && make migrate && make seed
```

---

## 5. Database Schema Design

### 5.1 Entity-Relationship Summary

The schema is organized into five domains: Identity (users, roles), Catalog (games, versions, tags), Community (forums, threads, posts), Engagement (reviews, likes, collections, follows), and System (notifications, reports, audit logs).

### 5.2 Core Tables

#### 5.2.1 Identity Domain

**users**

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PK, default gen_random_uuid() |
| username | VARCHAR(50) | UNIQUE, NOT NULL, indexed |
| email | VARCHAR(255) | UNIQUE, NOT NULL |
| password_hash | VARCHAR(255) | NOT NULL |
| display_name | VARCHAR(100) | |
| avatar_url | TEXT | |
| bio | TEXT | |
| website | VARCHAR(500) | |
| location | VARCHAR(100) | |
| role | ENUM('user','moderator','admin') | DEFAULT 'user' |
| is_active | BOOLEAN | DEFAULT true |
| is_verified | BOOLEAN | DEFAULT false |
| email_verified_at | TIMESTAMPTZ | |
| two_factor_enabled | BOOLEAN | DEFAULT false |
| two_factor_secret | VARCHAR(255) | |
| last_login_at | TIMESTAMPTZ | |
| created_at | TIMESTAMPTZ | DEFAULT now() |
| updated_at | TIMESTAMPTZ | DEFAULT now(), auto-update |

**user_profiles** (extended profile data, 1:1 with users)

| Column | Type | Constraints |
|---|---|---|
| user_id | UUID | PK, FK → users.id |
| social_discord | VARCHAR(100) | |
| social_twitter | VARCHAR(100) | |
| social_github | VARCHAR(100) | |
| patreon_url | VARCHAR(500) | |
| notification_preferences | JSONB | DEFAULT '{}' |
| privacy_settings | JSONB | DEFAULT '{}' |
| theme_preference | ENUM('light','dark','system') | DEFAULT 'system' |

#### 5.2.2 Catalog Domain

**games**

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PK |
| title | VARCHAR(255) | NOT NULL, indexed |
| slug | VARCHAR(300) | UNIQUE, NOT NULL, indexed |
| synopsis | TEXT | |
| plot | TEXT | |
| characters | TEXT | |
| walkthrough | TEXT | |
| engine_id | INT | FK → engines.id |
| author_id | UUID | FK → users.id (submitter) |
| original_pc_gender | ENUM('male','female','selectable','genderless','hermaphrodite') | |
| rating | ENUM('G','PG','R','X','XXX') | |
| development_status | ENUM('concept','demo','alpha','beta','complete','discontinued') | |
| is_free | BOOLEAN | DEFAULT true |
| has_purchasable_content | BOOLEAN | DEFAULT false |
| support_url | VARCHAR(500) | Patreon, Ko-fi, etc. |
| language | VARCHAR(50) | DEFAULT 'English' |
| play_online_url | VARCHAR(500) | |
| like_count | INT | DEFAULT 0 (denormalized counter) |
| review_count | INT | DEFAULT 0 |
| average_score | DECIMAL(3,2) | DEFAULT 0.00 |
| view_count | INT | DEFAULT 0 |
| play_count | INT | DEFAULT 0 |
| is_featured | BOOLEAN | DEFAULT false |
| is_approved | BOOLEAN | DEFAULT true |
| search_vector | TSVECTOR | GIN indexed, auto-updated |
| created_at | TIMESTAMPTZ | |
| updated_at | TIMESTAMPTZ | |

**game_versions**

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PK |
| game_id | UUID | FK → games.id, indexed |
| version_string | VARCHAR(50) | NOT NULL (e.g. "0.2.1") |
| changelog | TEXT | |
| release_date | DATE | |
| is_latest | BOOLEAN | DEFAULT true |
| created_at | TIMESTAMPTZ | |

**game_version_downloads** (multiple download mirrors per version)

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PK |
| version_id | UUID | FK → game_versions.id |
| url | TEXT | NOT NULL |
| label | VARCHAR(100) | e.g. "mega.nz", "mediafire", "Direct" |
| file_size_bytes | BIGINT | |
| platform | VARCHAR(50) | 'windows', 'mac', 'linux', 'browser', 'android' |

**game_screenshots**

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PK |
| game_id | UUID | FK → games.id |
| image_url | TEXT | NOT NULL |
| thumbnail_url | TEXT | |
| caption | VARCHAR(255) | |
| sort_order | INT | DEFAULT 0 |

**engines**

| Column | Type | Constraints |
|---|---|---|
| id | SERIAL | PK |
| name | VARCHAR(100) | UNIQUE, NOT NULL |
| slug | VARCHAR(120) | UNIQUE |
| game_count | INT | DEFAULT 0 (denormalized) |

**tags** (generic tag table for themes, genres, multimedia types, content warnings)

| Column | Type | Constraints |
|---|---|---|
| id | SERIAL | PK |
| name | VARCHAR(100) | NOT NULL |
| slug | VARCHAR(120) | UNIQUE |
| category | ENUM('genre','adult_theme','transformation','multimedia','content_warning','platform') | NOT NULL |
| description | TEXT | |

**game_tags** (many-to-many)

| Column | Type | Constraints |
|---|---|---|
| game_id | UUID | FK → games.id, PK |
| tag_id | INT | FK → tags.id, PK |

**game_authors** (many-to-many — games can have multiple authors/teams)

| Column | Type | Constraints |
|---|---|---|
| game_id | UUID | FK → games.id, PK |
| user_id | UUID | FK → users.id, PK |
| role | VARCHAR(50) | 'author', 'co-author', 'contributor' |

#### 5.2.3 Engagement Domain

**reviews**

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PK |
| game_id | UUID | FK → games.id |
| user_id | UUID | FK → users.id |
| version_reviewed | VARCHAR(50) | Snapshot of version at review time |
| score | SMALLINT | 1–10, NOT NULL |
| body | TEXT | NOT NULL |
| helpful_count | INT | DEFAULT 0 |
| not_helpful_count | INT | DEFAULT 0 |
| is_edited | BOOLEAN | DEFAULT false |
| created_at | TIMESTAMPTZ | |
| updated_at | TIMESTAMPTZ | |
| UNIQUE(game_id, user_id) | | One review per user per game |

**review_votes** (was this review helpful?)

| Column | Type | Constraints |
|---|---|---|
| review_id | UUID | FK → reviews.id, PK |
| user_id | UUID | FK → users.id, PK |
| is_helpful | BOOLEAN | NOT NULL |

**game_likes**

| Column | Type | Constraints |
|---|---|---|
| game_id | UUID | FK → games.id, PK |
| user_id | UUID | FK → users.id, PK |
| created_at | TIMESTAMPTZ | |

**collections** (user-curated game lists)

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PK |
| user_id | UUID | FK → users.id |
| name | VARCHAR(200) | NOT NULL |
| description | TEXT | |
| is_public | BOOLEAN | DEFAULT true |
| game_count | INT | DEFAULT 0 |
| created_at | TIMESTAMPTZ | |
| updated_at | TIMESTAMPTZ | |

**collection_games**

| Column | Type | Constraints |
|---|---|---|
| collection_id | UUID | FK → collections.id, PK |
| game_id | UUID | FK → games.id, PK |
| added_at | TIMESTAMPTZ | |
| sort_order | INT | DEFAULT 0 |
| note | TEXT | Personal note about why it's in the list |

**follows**

| Column | Type | Constraints |
|---|---|---|
| follower_id | UUID | FK → users.id, PK |
| following_id | UUID | FK → users.id, PK |
| created_at | TIMESTAMPTZ | |

**game_follows** (get notified when a game updates)

| Column | Type | Constraints |
|---|---|---|
| game_id | UUID | FK → games.id, PK |
| user_id | UUID | FK → users.id, PK |
| created_at | TIMESTAMPTZ | |

#### 5.2.4 Community Domain

**forum_categories**

| Column | Type | Constraints |
|---|---|---|
| id | SERIAL | PK |
| name | VARCHAR(200) | NOT NULL |
| slug | VARCHAR(220) | UNIQUE |
| description | TEXT | |
| sort_order | INT | |
| is_locked | BOOLEAN | DEFAULT false |
| parent_id | INT | FK → forum_categories.id (for subcategories) |
| thread_count | INT | DEFAULT 0 |
| post_count | INT | DEFAULT 0 |
| last_post_at | TIMESTAMPTZ | |

**threads**

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PK |
| forum_category_id | INT | FK → forum_categories.id |
| game_id | UUID | FK → games.id, NULLABLE (game discussion thread) |
| user_id | UUID | FK → users.id (author) |
| title | VARCHAR(300) | NOT NULL |
| slug | VARCHAR(350) | UNIQUE |
| is_pinned | BOOLEAN | DEFAULT false |
| is_locked | BOOLEAN | DEFAULT false |
| view_count | INT | DEFAULT 0 |
| post_count | INT | DEFAULT 0 |
| last_post_at | TIMESTAMPTZ | |
| created_at | TIMESTAMPTZ | |
| updated_at | TIMESTAMPTZ | |

**posts**

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PK |
| thread_id | UUID | FK → threads.id |
| user_id | UUID | FK → users.id |
| parent_id | UUID | FK → posts.id, NULLABLE (nested replies) |
| body | TEXT | NOT NULL |
| body_html | TEXT | Pre-rendered Markdown |
| is_edited | BOOLEAN | DEFAULT false |
| edited_at | TIMESTAMPTZ | |
| created_at | TIMESTAMPTZ | |

**chat_messages**

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PK |
| channel | VARCHAR(100) | NOT NULL (e.g. 'general', 'game:{id}') |
| user_id | UUID | FK → users.id |
| body | TEXT | NOT NULL |
| created_at | TIMESTAMPTZ | |

#### 5.2.5 System Domain

**notifications**

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PK |
| user_id | UUID | FK → users.id, indexed |
| type | VARCHAR(50) | 'review', 'reply', 'follow', 'game_update', 'mention', 'system' |
| title | VARCHAR(300) | |
| body | TEXT | |
| link | VARCHAR(500) | Deep link to relevant content |
| is_read | BOOLEAN | DEFAULT false |
| created_at | TIMESTAMPTZ | |

**reports** (content moderation)

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PK |
| reporter_id | UUID | FK → users.id |
| reason | ENUM('spam','harassment','inappropriate','copyright','other') | |
| description | TEXT | |
| target_type | VARCHAR(50) | 'game', 'review', 'post', 'user' |
| target_id | UUID | |
| status | ENUM('pending','reviewed','resolved','dismissed') | DEFAULT 'pending' |
| moderator_id | UUID | FK → users.id, NULLABLE |
| resolution_note | TEXT | |
| created_at | TIMESTAMPTZ | |
| resolved_at | TIMESTAMPTZ | |

**audit_logs**

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PK |
| user_id | UUID | FK → users.id |
| action | VARCHAR(100) | 'game.create', 'user.ban', 'review.delete', etc. |
| target_type | VARCHAR(50) | |
| target_id | UUID | |
| metadata | JSONB | Before/after snapshots, IP, user-agent |
| ip_address | INET | |
| created_at | TIMESTAMPTZ | |

### 5.3 Indexes Strategy

- `games.search_vector` — GIN index for PostgreSQL full-text search (fallback to Meilisearch for user-facing search).
- `games(slug)` — Unique B-tree for URL lookups.
- `games(author_id, created_at DESC)` — Composite for "my games" queries.
- `reviews(game_id, created_at DESC)` — Reviews listing.
- `game_likes(game_id)` — Counting likes.
- `notifications(user_id, is_read, created_at DESC)` — Notification bell badge count + list.
- `threads(forum_category_id, is_pinned DESC, last_post_at DESC)` — Forum thread ordering.
- `posts(thread_id, created_at)` — Chronological post listing.

---

## 6. Backend Requirements (FastAPI)

### 6.1 Application Structure Principles

- **Layered architecture**: Routes (thin) → Services (business logic) → Repositories (data access) → Models (ORM).
- **Async throughout**: All database queries use `async/await` with `asyncpg` and SQLAlchemy 2.0 async sessions.
- **Dependency injection**: FastAPI `Depends()` for database sessions, current user, permissions, pagination.
- **API versioning**: All routes live under `/api/v1/...` to allow future breaking changes under `/api/v2/`.
- **Consistent error responses**: Every error returns `{ "detail": "...", "error_code": "UNIQUE_CODE", "status": 4xx }`.

### 6.2 Authentication & Authorization

**Endpoints:**

```
POST   /api/v1/auth/register           # Create account
POST   /api/v1/auth/login               # Returns access + refresh tokens
POST   /api/v1/auth/refresh              # Exchange refresh token for new access token
POST   /api/v1/auth/logout               # Blacklist tokens in Redis
POST   /api/v1/auth/forgot-password      # Send reset email
POST   /api/v1/auth/reset-password       # Consume reset token
POST   /api/v1/auth/verify-email/{token} # Email verification
POST   /api/v1/auth/2fa/setup            # Generate TOTP secret + QR
POST   /api/v1/auth/2fa/verify           # Confirm TOTP setup
POST   /api/v1/auth/2fa/disable          # Turn off 2FA
GET    /api/v1/auth/oauth/{provider}     # OAuth2 redirect (Google, GitHub, Discord)
GET    /api/v1/auth/oauth/{provider}/callback
```

**Implementation details:**

- Passwords hashed with **bcrypt** (cost factor 12).
- JWT access tokens (30 min) and refresh tokens (7 days) stored as HTTP-only cookies (with SameSite=Lax) and also returned in the response body for mobile clients.
- Token blacklist stored in Redis with TTL matching the token's remaining lifetime.
- Role-based access control (RBAC): `user`, `moderator`, `admin`. Enforced via a `require_role()` dependency.
- Rate limiting on auth endpoints: 5 login attempts per 15 minutes per IP (Redis counters).
- OAuth2 via `authlib` supporting Google, GitHub, and Discord.

### 6.3 User Management

```
GET    /api/v1/users/me                  # Current user profile
PUT    /api/v1/users/me                  # Update profile
DELETE /api/v1/users/me                  # Soft delete account
GET    /api/v1/users/{username}          # Public profile
GET    /api/v1/users/{username}/games    # Games by this user
GET    /api/v1/users/{username}/reviews  # Reviews by this user
GET    /api/v1/users/{username}/collections  # Public collections
POST   /api/v1/users/{username}/follow   # Follow a user
DELETE /api/v1/users/{username}/follow   # Unfollow
GET    /api/v1/users/{username}/followers
GET    /api/v1/users/{username}/following
PUT    /api/v1/users/me/avatar           # Upload avatar (multipart)
PUT    /api/v1/users/me/password         # Change password (requires current)
PUT    /api/v1/users/me/notifications    # Update notification preferences
```

### 6.4 Game Management

```
GET    /api/v1/games                     # List/search with pagination + filters
GET    /api/v1/games/{slug}              # Game detail (SSR-friendly)
POST   /api/v1/games                     # Submit new game (authenticated)
PUT    /api/v1/games/{slug}              # Update game metadata (author/admin)
DELETE /api/v1/games/{slug}              # Soft delete (admin)
POST   /api/v1/games/{slug}/versions     # Add new version (author)
GET    /api/v1/games/{slug}/versions     # Version history
PUT    /api/v1/games/{slug}/versions/{id} # Edit version info
POST   /api/v1/games/{slug}/screenshots  # Upload screenshots (presigned URL flow)
DELETE /api/v1/games/{slug}/screenshots/{id}
POST   /api/v1/games/{slug}/like         # Like/unlike toggle
GET    /api/v1/games/{slug}/similar      # Recommendation engine results
POST   /api/v1/games/{slug}/follow       # Follow game for update notifications
DELETE /api/v1/games/{slug}/follow
GET    /api/v1/games/featured            # Featured games
GET    /api/v1/games/trending            # Trending this week (by views + likes)
GET    /api/v1/games/recent              # Recently submitted
GET    /api/v1/games/recently-updated    # Recently updated
```

**Query parameters for `GET /api/v1/games`:**

| Parameter | Type | Description |
|---|---|---|
| q | string | Full-text search query |
| engine | string[] | Filter by engine slug(s) |
| status | string[] | Development status |
| genre | string[] | Genre tag slugs |
| adult_theme | string[] | Adult theme tag slugs |
| transformation | string[] | Transformation theme tag slugs |
| multimedia | string[] | Multimedia type tag slugs |
| content_warning | string[] | Content warning tag slugs |
| rating | string[] | Content rating |
| pc_gender | string[] | Original PC gender |
| author | string | Author username |
| has_play_online | boolean | Filter play-online available |
| min_likes | int | Minimum like count |
| sort | string | 'newest', 'updated', 'rating', 'likes', 'title', 'trending' |
| page | int | Page number (default 1) |
| per_page | int | Items per page (default 24, max 100) |

### 6.5 Review System

```
GET    /api/v1/games/{slug}/reviews      # List reviews for a game
POST   /api/v1/games/{slug}/reviews      # Submit a review (one per user per game)
PUT    /api/v1/reviews/{id}              # Edit own review
DELETE /api/v1/reviews/{id}              # Delete own review
POST   /api/v1/reviews/{id}/vote         # Vote helpful/not helpful
GET    /api/v1/reviews/recent            # Global recent reviews feed
```

**Business rules:**
- Users can only review a game once. Subsequent submissions return `409 Conflict`.
- Reviews capture the version string at submission time for context.
- Editing a review marks it as `is_edited=true` and updates `updated_at`.
- Review score is 1–10. The game's `average_score` is recalculated asynchronously via Celery.
- Users cannot vote on their own reviews.

### 6.6 Forum & Threads

```
GET    /api/v1/forums                    # List forum categories
GET    /api/v1/forums/{slug}             # List threads in a category
POST   /api/v1/forums/{slug}/threads     # Create thread
GET    /api/v1/threads/{slug}            # Thread detail with posts
POST   /api/v1/threads/{slug}/posts      # Reply to thread
PUT    /api/v1/posts/{id}                # Edit own post
DELETE /api/v1/posts/{id}                # Delete own post (or moderator)
POST   /api/v1/threads/{slug}/lock       # Lock thread (moderator)
POST   /api/v1/threads/{slug}/pin        # Pin thread (moderator)
```

**Game discussion threads:** When a game is created, an auto-linked discussion thread is created in the "Game Discussions" forum category with `game_id` set. The game detail page links directly to it.

**Markdown support:** Post bodies are stored as raw Markdown. A pre-rendered `body_html` column is populated on save using a server-side Markdown renderer with sanitization (no raw HTML, no script tags). The frontend renders `body_html` directly.

### 6.7 Real-Time Chat & WebSockets

```
WS     /ws/chat/{channel}               # Join a chat channel
WS     /ws/notifications                 # User-specific notification stream
```

**Chat implementation:**
- Channels: `general` (global), `game:{game_id}` (per-game chat).
- Messages are persisted to `chat_messages` table.
- Fan-out via Redis pub/sub: FastAPI publishes to a Redis channel, all connected WebSocket instances subscribe and push to their clients.
- Last 50 messages are loaded on channel join (HTTP fetch, then WebSocket for new ones).
- Rate limit: 1 message per 2 seconds per user (Redis counter).

**Notification stream:**
- Authenticated WebSocket connection. On new notification (triggered by Celery task), the backend publishes to `notifications:{user_id}` Redis channel, and the user's connected WebSocket pushes the event.

### 6.8 Search

```
GET    /api/v1/search                    # Proxies to Meilisearch
GET    /api/v1/search/suggestions        # Autocomplete suggestions
```

**Meilisearch index configuration:**
- Index name: `games`
- Searchable attributes: `title`, `synopsis`, `plot`, `author_names`, `engine_name`, `tag_names`
- Filterable attributes: `engine_slug`, `development_status`, `rating`, `tag_slugs`, `original_pc_gender`, `has_play_online`, `like_count`
- Sortable attributes: `created_at`, `updated_at`, `like_count`, `average_score`, `title`
- Ranking rules: `words`, `typo`, `proximity`, `attribute`, `sort`, `exactness`
- Sync: A Celery task runs on every game create/update/delete to upsert or remove the document from Meilisearch.

### 6.9 Notifications

```
GET    /api/v1/notifications             # Paginated notification list
GET    /api/v1/notifications/unread-count
POST   /api/v1/notifications/mark-read   # Mark specific or all as read
DELETE /api/v1/notifications/{id}
```

**Notification triggers (via Celery tasks):**

| Event | Notified User(s) |
|---|---|
| New review on your game | Game author |
| Reply to your thread | Thread author |
| Reply to your post | Post author |
| New follower | Followed user |
| Game you follow was updated | All game followers |
| Your game was featured | Game author |
| Mention (@username) in post | Mentioned user |
| Report resolved | Reporter |
| Account actions (ban, etc.) | Affected user |

### 6.10 Collections

```
GET    /api/v1/collections               # Public collections feed
POST   /api/v1/collections               # Create collection
GET    /api/v1/collections/{id}          # Collection detail with games
PUT    /api/v1/collections/{id}          # Update name/description/visibility
DELETE /api/v1/collections/{id}
POST   /api/v1/collections/{id}/games    # Add game to collection
DELETE /api/v1/collections/{id}/games/{game_id}
PUT    /api/v1/collections/{id}/games/reorder  # Reorder games
```

### 6.11 Admin & Moderation

```
GET    /api/v1/admin/dashboard           # Aggregate stats
GET    /api/v1/admin/users               # User list with filters
PUT    /api/v1/admin/users/{id}/role     # Change user role
POST   /api/v1/admin/users/{id}/ban      # Ban user
POST   /api/v1/admin/users/{id}/unban
GET    /api/v1/admin/games/pending       # Games awaiting approval
POST   /api/v1/admin/games/{id}/approve
POST   /api/v1/admin/games/{id}/reject
GET    /api/v1/admin/reports             # Moderation queue
PUT    /api/v1/admin/reports/{id}        # Resolve/dismiss report
GET    /api/v1/admin/audit-log           # System audit trail
DELETE /api/v1/admin/reviews/{id}        # Force-delete review
DELETE /api/v1/admin/posts/{id}          # Force-delete post
POST   /api/v1/admin/tags               # Create tag
PUT    /api/v1/admin/tags/{id}           # Edit tag
POST   /api/v1/admin/engines            # Create engine
PUT    /api/v1/admin/engines/{id}        # Edit engine
POST   /api/v1/admin/forum-categories   # Create forum category
PUT    /api/v1/admin/forum-categories/{id}
```

### 6.12 File Upload Flow

All file uploads use a **presigned URL pattern** to keep binary data off the API server:

1. Client calls `POST /api/v1/uploads/presign` with `{ filename, content_type, purpose }`.
2. Backend validates the request (file type whitelist, size limits per purpose) and generates a presigned PUT URL from MinIO.
3. Client uploads the file directly to MinIO using the presigned URL.
4. Client confirms the upload by calling the relevant endpoint (e.g., `POST /api/v1/games/{slug}/screenshots`) with the object key.
5. Backend verifies the object exists in MinIO, generates a thumbnail (Celery task for images), and records the metadata.

**Upload limits by purpose:**

| Purpose | Max Size | Allowed Types |
|---|---|---|
| Avatar | 5 MB | JPEG, PNG, WebP, GIF |
| Screenshot | 10 MB | JPEG, PNG, WebP, GIF |
| Game file | 2 GB | ZIP, RAR, 7Z, EXE, APK |
| Forum attachment | 20 MB | JPEG, PNG, WebP, GIF, PDF, ZIP |

---

## 7. Frontend Requirements (Next.js)

### 7.1 Tech Stack Details

| Library | Purpose |
|---|---|
| **Next.js 14+** | Framework (App Router, SSR, ISR) |
| **TypeScript** | Type safety |
| **Tailwind CSS 3** | Utility-first styling |
| **shadcn/ui** | Accessible component primitives |
| **TanStack Query (React Query)** | Server state management, caching, pagination |
| **Zustand** | Client state (auth, notifications, chat) |
| **React Hook Form + Zod** | Form handling and validation |
| **Framer Motion** | Page transitions, micro-interactions |
| **next-themes** | Dark/light/system mode |
| **date-fns** | Date formatting |
| **react-markdown + rehype-sanitize** | Markdown rendering in posts/reviews |
| **Playwright** | End-to-end testing |

### 7.2 Design System & Visual Identity

**Palette (CSS custom properties for theme switching):**

```
Light mode:
  --background:     #FAFAFA (warm off-white)
  --surface:        #FFFFFF
  --surface-raised: #F4F4F5 (zinc-100)
  --border:         #E4E4E7 (zinc-200)
  --text-primary:   #18181B (zinc-900)
  --text-secondary: #71717A (zinc-500)
  --accent:         #6366F1 (indigo-500)
  --accent-hover:   #4F46E5 (indigo-600)
  --success:        #22C55E (green-500)
  --warning:        #F59E0B (amber-500)
  --error:          #EF4444 (red-500)

Dark mode:
  --background:     #09090B (zinc-950)
  --surface:        #18181B (zinc-900)
  --surface-raised: #27272A (zinc-800)
  --border:         #3F3F46 (zinc-700)
  --text-primary:   #FAFAFA (zinc-50)
  --text-secondary: #A1A1AA (zinc-400)
  --accent:         #818CF8 (indigo-400)
  --accent-hover:   #6366F1 (indigo-500)
```

**Typography:**
- Display/Headings: **Inter** (variable weight, crisp on screens)
- Body: **Inter** (consistent family, weight variation for hierarchy)
- Monospace (code blocks, version strings): **JetBrains Mono**
- Type scale: 12/14/16/18/20/24/30/36/48px with consistent line heights

**Layout:**
- Max content width: 1280px, centered.
- Responsive breakpoints: `sm` 640px, `md` 768px, `lg` 1024px, `xl` 1280px.
- Sidebar (filters, game metadata): 300px fixed on desktop, slides in as drawer on mobile.
- Card grid: 4 columns on xl, 3 on lg, 2 on md, 1 on sm.

**Signature element:** A subtle gradient "glow" behind the hero section's featured game cards, using the accent color at low opacity, creating a focal point that shifts with dark/light mode.

### 7.3 Page-by-Page Specifications

#### 7.3.1 Homepage (`/`)

**Layout:** Full-width hero → three content sections → footer.

**Sections:**
1. **Hero area**: Large headline ("Discover, Share & Discuss Games"), search bar (prominent, centered), and a "Browse All" CTA. Below: quick-stat counters (Total Games, Total Reviews, Engines, Online Plays) with animated count-up on scroll.
2. **Featured Community Favorites**: Horizontal scrollable card row (12 items). Cards show game title, author, engine badge, like count, development status pill.
3. **Trending This Week**: Grid of 12 cards. Trending score calculated as `(likes_this_week * 2) + views_this_week + (reviews_this_week * 5)`.
4. **Recent Submissions**: Grid of 12 cards sorted by `created_at DESC`.
5. **Recent Updates**: Grid of 12 cards sorted by `updated_at DESC` where a new version was added.
6. **Latest Reviews**: List of 12 review excerpts showing game title, reviewer name, score badge, date, and first 100 characters of the review body.

#### 7.3.2 Game Browse/Search (`/games`)

**Layout:** Filter sidebar (left) + results grid (right).

**Filter panel:**
- Text search input (debounced, hits Meilisearch).
- "Search in" selector: Title, Synopsis, Plot, Characters, Walkthrough, or All.
- Collapsible filter groups (each group is an accordion):
  - Development Status (checkboxes with include/exclude toggle, matching reference site's +/- pattern).
  - Engine (searchable multi-select).
  - Genre Tags (checkboxes).
  - Adult Themes (checkboxes).
  - Transformation Themes (checkboxes).
  - Multimedia (checkboxes).
  - Content Warnings (checkboxes).
  - Content Rating (G, PG, R, X, XXX).
  - Original PC Gender.
  - Likes range (min–max slider).
  - Play Online available (toggle).
  - Author (text search).
- Active filters displayed as removable chips above the results.
- "Clear all filters" button.

**Results:**
- Sort dropdown: Newest, Recently Updated, Highest Rated, Most Liked, Title A-Z, Trending.
- View toggle: Grid (cards) / List (compact rows).
- Pagination: Page numbers + "Load more" infinite scroll option.
- Result count: "Showing 1–24 of 1,847 games".

#### 7.3.3 Game Detail (`/games/[slug]`)

**Layout:** Two-column — main content (left, ~70%) + metadata sidebar (right, ~30%).

**Main content area:**
- **Header**: Game title (h1), author link(s), engine badge, status pill, rating badge.
- **Screenshot gallery**: Lightbox carousel if screenshots exist. Thumbnails in a horizontal strip below.
- **Tab panel** (Synopsis / Plot / Characters / Walkthrough / Changelog): Default to Synopsis. All support Markdown rendering.
- **Actions**: Like button (with count), Follow button, Report button, Share dropdown (copy link, Twitter, Reddit).
- **"Users who liked this also liked..."**: Horizontal scroll row of 6 similar games (collaborative filtering or tag overlap).
- **Reviews section**: Paginated list of reviews, each showing username, avatar, score badge, version reviewed, date, body, and helpful/not-helpful vote buttons. "Write a review" button (opens inline form or modal).
- **Discussion link**: Prominent button/link to the auto-created discussion thread.

**Metadata sidebar:**
- Engine (linked to browse-by-engine)
- Base game cost (Free / $X)
- Purchasable content (Yes/No)
- Content rating
- Language
- Release date
- Last update date
- Latest version string
- Development status
- Like count
- Original PC gender
- Theme tags (each linked to search with that filter)
- Multimedia tags
- Content warnings
- Support link (Patreon, Ko-fi, etc.)
- **Download section**: Grouped by version. Each version shows version string, release date, and download mirror buttons. Latest version highlighted.
- **Play Online button** (if `play_online_url` exists)

#### 7.3.4 Game Submission (`/games/submit`)

**Access:** Authenticated users only. Redirects to login if not authenticated.

**Form sections (multi-step wizard or single long form with sidebar navigation):**

1. **Basic info**: Title, engine (dropdown), development status, content rating, original PC gender, language, is_free, has_purchasable_content, support_url.
2. **Description**: Synopsis (rich Markdown editor with preview), Plot, Characters, Walkthrough — each with a Markdown editor.
3. **Tags**: Multi-select for genre, adult themes, transformation themes, multimedia, content warnings. Presented as searchable checkbox groups.
4. **Media**: Screenshot upload zone (drag-and-drop, up to 10 images). Reorderable via drag.
5. **Downloads**: Add version entry — version string, changelog, release date. Add download mirrors (URL, label, platform, file size). Optionally upload game file directly to MinIO.
6. **Play Online**: Optional URL for browser-playable version.
7. **Review & Submit**: Summary of all entered data. Submit button.

**Validation:** All fields validated client-side (Zod) and server-side (Pydantic). Title uniqueness checked via debounced API call.

#### 7.3.5 User Profile (`/profile/[username]`)

**Public view:**
- Avatar, display name, username, bio, member since date, location, website, social links.
- Stats: Games submitted, reviews written, forum posts, followers, following.
- Tabs: Games (grid of their submissions), Reviews (list), Collections (public), Activity (feed of recent actions).

**Own profile (additional):**
- "Edit profile" button → settings page.
- Private stats: Total likes received across all games, notification count.

#### 7.3.6 Dashboard (`/dashboard`)

**Authenticated users only.** A personalized landing page.

- **Activity feed**: Aggregated timeline of things relevant to the user — new reviews on their games, replies to their threads, games they follow being updated, new followers.
- **My Games**: Quick list of their submitted games with edit/update buttons and per-game stats (views, likes, reviews).
- **My Reviews**: List of reviews they've written.
- **Notifications**: Full notification list (also accessible from the bell icon anywhere).

#### 7.3.7 Community Forums (`/community`)

**Forum index:** List of forum categories with description, thread count, post count, and "last post" preview. Categories can have subcategories (nested one level).

**Category view (`/community/[forumSlug]`):** Thread list showing title, author, reply count, view count, last reply preview, pinned badge, locked badge. Pagination. "New thread" button.

**Thread view (`/community/[forumSlug]/[threadSlug]`):** Original post at top, followed by chronological replies. Each post shows author avatar, username, role badge, post date, body (rendered Markdown), and action buttons (reply, quote, edit, delete, report). Nested replies shown with indentation (one level of nesting). Pagination for long threads.

**Post editor:** Markdown textarea with toolbar (bold, italic, link, image, code block, quote, spoiler). Live preview toggle.

#### 7.3.8 Chat (`/chat`)

**Layout:** Channel list (left sidebar) + message area (center) + online users list (right sidebar, collapsible).

**Features:**
- `#general` channel always visible.
- Game-specific channels created automatically when a game has its first chat message.
- Message input at bottom with send button and character limit (500).
- Messages show avatar, username, timestamp, and body.
- Auto-scroll to latest message. "New messages" indicator when scrolled up.
- Online presence indicators (green dot).
- Message history loaded via API, new messages via WebSocket.

#### 7.3.9 Admin Panel (`/admin`)

**Access:** `admin` role only. Separate layout with admin sidebar navigation.

**Dashboard:** Cards showing key metrics — new users (today/week/month), new games, new reviews, pending reports, active users. Line charts for user registrations and game submissions over time (using Recharts).

**Users management:** Searchable table with columns: avatar, username, email, role, status, join date, last login. Actions: view profile, change role, ban/unban.

**Games management:** Table with title, author, status, approval status, created date. Actions: approve, reject, feature/unfeature, delete.

**Reports queue:** Table showing reported content with reporter, reason, target, status. Inline preview of reported content. Actions: resolve (with note), dismiss, ban reporter.

**Audit log:** Searchable, filterable table of all admin and moderation actions.

### 7.4 Shared UX Patterns

**Loading states:** Skeleton placeholders matching the layout of the content being loaded (never spinners alone). TanStack Query's `isLoading` and `isFetching` states distinguished — stale data shown while refetching.

**Empty states:** Illustrated empty state with a message and a CTA. Examples: "No games match your filters. Try broadening your search." / "You haven't submitted any games yet. Share your first game!"

**Error handling:** Toast notifications for non-blocking errors (failed like, network glitch). Full-page error boundary for critical failures. Retry buttons where applicable.

**Optimistic updates:** Like/unlike, follow/unfollow, and review helpfulness votes update the UI immediately and roll back on failure.

**Infinite scroll vs. pagination:** Search results use pagination (SEO). Forum threads use pagination. Activity feeds and chat use infinite scroll.

**Accessibility:** All interactive elements are keyboard-navigable. ARIA labels on icon-only buttons. Focus management on modals and drawers. Color contrast ratios meet WCAG 2.1 AA. Reduced motion respected via `prefers-reduced-motion`.

---

## 8. Feature Specifications

### 8.1 Recommendation Engine ("Users who liked this also liked...")

**Algorithm:** Collaborative filtering using the `game_likes` table. For a given game G, find all users who liked G, then rank other games by how many of those users also liked them, excluding games the viewing user has already liked. Fallback to tag-similarity (Jaccard index on tag sets) when like data is sparse (<10 likes).

**Implementation:** Precomputed nightly by a Celery beat task. Results stored in a `game_recommendations` table (game_id, recommended_game_id, score, rank). Cached in Redis with a 24-hour TTL. Top 6 returned per game.

### 8.2 Trending Score Calculation

**Formula:** `trending_score = (likes_7d * 3) + (views_7d * 0.1) + (reviews_7d * 10) + (new_version_bonus * 20)`

Where `new_version_bonus` is 1 if the game received a version update in the last 7 days, 0 otherwise. Recalculated hourly by a Celery beat task. Stored in a `game_trending_scores` table or Redis sorted set.

### 8.3 Content Moderation Pipeline

1. User clicks "Report" and selects a reason + optional description.
2. Report is created with `status=pending`.
3. Moderators see pending reports in `/admin/reports`, sorted by created date.
4. Moderator reviews the content (inline preview) and takes action:
   - **Resolve**: Remove content + optionally warn/ban the offender. Status → `resolved`.
   - **Dismiss**: Content stays, report is marked `dismissed`.
5. Reporter receives a notification that their report was reviewed.
6. All moderation actions are recorded in `audit_logs`.

**Automatic moderation (future):** Integrate a profanity filter on game submissions, reviews, and forum posts. Flag content for review rather than auto-rejecting.

### 8.4 Email Notifications

Sent via Celery tasks through SMTP. Templates rendered with Jinja2. Users can configure which notifications generate emails in their notification preferences (JSONB field).

**Email types:**
- Welcome / email verification
- Password reset
- New review on your game
- Game you follow was updated
- Weekly digest (opt-in): summary of trending games, new submissions in followed engines/tags
- Account security alerts (login from new device, password changed)

### 8.5 SEO Strategy

- Game detail pages are server-side rendered with full metadata (Open Graph, Twitter cards, JSON-LD structured data for "VideoGame" schema).
- Dynamic `sitemap.xml` generated from all public game slugs.
- Canonical URLs on all pages.
- `robots.txt` allowing crawling of public pages, blocking admin and API routes.
- Page titles follow pattern: `{Game Title} by {Author} — GameVault` or `{Page Name} — GameVault`.

---

## 9. Non-Functional Requirements

### 9.1 Performance Targets

| Metric | Target |
|---|---|
| Time to First Byte (TTFB) | < 200ms (SSR pages) |
| Largest Contentful Paint (LCP) | < 2.5s |
| First Input Delay (FID) | < 100ms |
| Cumulative Layout Shift (CLS) | < 0.1 |
| API response time (p95) | < 300ms |
| Search response time (p95) | < 100ms (Meilisearch) |
| WebSocket message latency | < 50ms |
| Database query time (p95) | < 50ms |

### 9.2 Scalability

- Backend: Stateless FastAPI instances behind Traefik. Scale horizontally by adding more `backend` containers.
- Celery: Add more worker containers to handle increased task volume.
- PostgreSQL: Read replicas for search/browse queries (future). Connection pooling via PgBouncer.
- Redis: Sentinel or Redis Cluster for HA (production).
- MinIO: Distributed mode across multiple nodes (production).
- Meilisearch: Single node is sufficient for <1M documents.

### 9.3 Security

| Concern | Mitigation |
|---|---|
| XSS | Markdown rendering with rehype-sanitize; CSP headers |
| CSRF | SameSite cookies; CSRF tokens for state-changing requests |
| SQL injection | SQLAlchemy ORM with parameterized queries; no raw SQL |
| Rate limiting | Redis-backed sliding window (per IP + per user) |
| File uploads | Presigned URLs (never through API server); MIME validation; virus scanning (ClamAV, future) |
| Secrets | Environment variables only; never committed to VCS |
| Dependency vulnerabilities | Dependabot / Snyk in CI |
| Data at rest | PostgreSQL encryption (TDE, production); MinIO server-side encryption |
| Transport | TLS everywhere (Traefik handles termination) |
| Password storage | bcrypt with cost factor 12 |
| 2FA | TOTP-based (Google Authenticator, Authy compatible) |

### 9.4 Monitoring & Observability

- **Logging**: Structured JSON logs (structlog for Python, pino for Node). Log levels: DEBUG (dev), INFO (prod).
- **Metrics**: Prometheus endpoint on backend (`/metrics`). Grafana dashboards for request rate, error rate, latency histograms, database pool stats, Redis memory, Celery queue depth.
- **Tracing**: OpenTelemetry instrumentation on backend. Jaeger for distributed trace visualization (dev).
- **Health checks**: `/api/v1/health` endpoint returning status of all dependencies (DB, Redis, Meilisearch, MinIO). Used by Docker health checks and Traefik.
- **Error tracking**: Sentry integration (backend + frontend).

### 9.5 Testing Strategy

| Layer | Tool | Coverage Target |
|---|---|---|
| Backend unit tests | pytest + pytest-asyncio | 80%+ |
| Backend integration tests | pytest + httpx (TestClient) | All API endpoints |
| Frontend component tests | Vitest + Testing Library | Critical components |
| End-to-end tests | Playwright | Core user flows (register, submit game, review, search) |
| Load testing | Locust | Key endpoints under 100 concurrent users |
| API contract tests | schemathesis (fuzz OpenAPI schema) | All endpoints |

---

## 10. Deployment & DevOps

### 10.1 Docker Image Strategy

**Backend Dockerfile (multi-stage):**
```
Stage 1 "builder": python:3.12-slim, install dependencies via uv/pip
Stage 2 "development": copy source, run uvicorn with --reload
Stage 3 "production": copy only installed packages + source, run uvicorn with --workers 4
```

**Frontend Dockerfile (multi-stage):**
```
Stage 1 "deps": node:20-alpine, install npm packages
Stage 2 "development": copy source, run next dev
Stage 3 "builder": next build (generates .next/)
Stage 4 "production": node:20-alpine, copy .next/standalone, run node server.js
```

### 10.2 CI/CD Pipeline (GitHub Actions)

```
on: [push, pull_request]

jobs:
  lint:        # ruff (Python), eslint + prettier (TypeScript)
  type-check:  # mypy (Python), tsc --noEmit (TypeScript)
  test-backend:
    services: [postgres, redis]
    steps: alembic upgrade head → pytest --cov
  test-frontend:
    steps: npm test → npx playwright test
  build:
    steps: docker compose build
  deploy:      # On main branch merge only
    steps: push images to container registry → deploy to staging
```

### 10.3 Backup Strategy

- **PostgreSQL**: `pg_dump` via cron (daily full, hourly WAL archiving to MinIO).
- **MinIO**: Cross-region replication (production) or periodic `mc mirror` to backup storage.
- **Redis**: AOF persistence enabled; periodic RDB snapshots.
- **Meilisearch**: Snapshots via API scheduled daily.

### 10.4 Production Considerations

- Run Traefik with Let's Encrypt for automatic TLS certificate renewal.
- Use a managed PostgreSQL service (AWS RDS, DigitalOcean, etc.) for production reliability.
- Put MinIO behind a CDN (CloudFront, Cloudflare) for static asset delivery.
- Set `APP_ENV=production` to disable debug modes, tighten CORS, and enable security headers.
- Configure log rotation and retention policies.

---

## 11. Appendix: Suggested Enhancements Beyond Reference

These features go beyond what the reference site offers and represent opportunities to differentiate GameVault.

### 11.1 Game Jams / Contests

A built-in system for organizing community game jams. Admins or designated users can create a jam with a theme, deadline, and rules. Users submit entries (linked to game listings). Community voting during a defined voting period. Leaderboard and winner announcements. Past jams archived with results.

### 11.2 Developer Journals

Game authors can write development blog posts attached to their game listing. Each journal entry has a title, body (Markdown), and optional screenshots. Followers of the game receive notifications for new journal entries. Displayed on the game detail page as a "Dev Journal" tab.

### 11.3 Achievement System

Gamification badges awarded for community participation. Examples: "First Game" (submitted first game), "Prolific Author" (10+ games), "Critic" (50+ reviews), "Community Pillar" (1000+ forum posts), "Trendsetter" (game trending #1). Badges displayed on user profiles.

### 11.4 Game Comparison Tool

Users select 2–4 games and see a side-by-side comparison table of their metadata: engine, status, rating, like count, review average, themes, last update, etc. Useful for deciding between similar games.

### 11.5 Wishlists

A lightweight "Want to Play" list. Single-click bookmark on any game. Separate from collections (which are curated and public by default). Wishlist is private. Users can get email digests when wishlisted games receive updates.

### 11.6 Play History & Ratings Tracker

Users can mark games as "Playing", "Played", "On Hold", "Dropped" (similar to MyAnimeList). Personal notes and play dates. This data feeds into the recommendation engine.

### 11.7 Public API for Developers

A documented REST API (the same API the frontend uses) available for third-party developers. API keys managed in user settings. Rate-limited to 100 requests/minute. OpenAPI spec available at `/api/v1/docs`. Use cases: Discord bots that report new games, personal website widgets showing a user's game list, community tools.

### 11.8 Localization (i18n)

Multi-language support for the UI using `next-intl`. Initial languages: English, Spanish, Portuguese, Japanese, Chinese (Simplified). Game content itself remains in its original language, but UI chrome (buttons, labels, navigation) is translated. Language preference stored in user profile and detected via browser `Accept-Language` header for anonymous users.

### 11.9 RSS Feeds

Standard RSS/Atom feeds for: new game submissions, recent game updates, latest reviews, specific engine new releases, specific author new releases. URLs like `/feed/games/new`, `/feed/games/updated`, `/feed/reviews`, `/feed/engine/{slug}`, `/feed/author/{username}`.

### 11.10 Embed Widgets

Embeddable HTML snippets that third parties can place on their own websites. Widget types: "Game card" (shows title, cover, rating, link), "Author badge" (shows user profile, game count), "Collection showcase" (scrollable game strip). Generated from `/embed/game/{slug}`, served as lightweight standalone HTML+CSS.

### 11.11 Advanced Analytics for Authors

Authenticated game authors see a private analytics tab on their game pages: daily/weekly/monthly views, download counts per mirror, like/unlike trends, review sentiment summary, geographic distribution of visitors (country-level, via anonymized IP geolocation), referrer breakdown (where traffic comes from).

### 11.12 Content Version Diffing

When a game's synopsis, plot, or other text fields are edited, store the previous version. Show a "History" link with diffs (like Wikipedia's edit history). Useful for moderation and transparency.

---

## End of Requirements Document

This document serves as the comprehensive blueprint for the GameVault platform. Each section should be refined and expanded into implementation tickets during sprint planning. The Docker Compose setup ensures any contributor can run the full stack locally with a single `make up` command.
