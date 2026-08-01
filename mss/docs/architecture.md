# GameVault Architecture

This document describes the key architectural decisions behind GameVault and the rationale for each choice.

---

## Why FastAPI + Next.js

### FastAPI (Backend)

| Factor | Decision |
|---|---|
| Async I/O | Native `async/await` for database, Redis, and HTTP calls — critical for WebSocket chat and high-concurrency API |
| Type safety | Pydantic v2 models provide runtime validation and auto-generated OpenAPI schemas |
| Performance | One of the fastest Python frameworks; comparable to Node.js and Go for I/O-bound workloads |
| WebSockets | Built-in WebSocket support without additional dependencies |
| Ecosystem | Strong async SQLAlchemy 2.0, Celery, and Redis library support |
| Developer experience | Auto-generated interactive API docs at `/api/v1/docs` |

**Alternatives considered:**

- **Django + DRF**: Mature but synchronous by default; async support is still maturing. Heavier framework for an API-only backend.
- **Node.js (Express/Fastify)**: Good performance but splitting the stack across two JS runtimes (Node frontend + Node backend) offers less benefit than Python for data-heavy operations and Celery integration.
- **Go (Fiber/Gin)**: Excellent performance but slower development velocity for a feature-rich community platform.

### Next.js (Frontend)

| Factor | Decision |
|---|---|
| SSR/SSG | Server-side rendering for game detail pages — critical for SEO and social sharing (Open Graph tags) |
| App Router | React Server Components reduce client-side JavaScript for browse/listing pages |
| File-based routing | Intuitive mapping from URL structure to page components |
| Ecosystem | Largest React meta-framework; excellent TypeScript, Tailwind, and deployment tooling |
| API integration | Server Components can fetch from FastAPI during SSR, eliminating client-side waterfalls |

**Alternatives considered:**

- **SPA (Vite + React)**: Poor SEO for game pages without additional SSR infrastructure.
- **Remix**: Strong SSR model but smaller ecosystem and fewer UI component libraries.
- **SvelteKit**: Excellent performance but smaller talent pool and component ecosystem.

---

## Database Design Rationale

### PostgreSQL as Primary Store

PostgreSQL 16 was chosen for:

- **Relational integrity**: Games, users, reviews, and forums have complex many-to-many relationships that map naturally to relational tables.
- **JSONB columns**: Flexible storage for notification preferences, privacy settings, and audit log metadata without sacrificing queryability.
- **Full-text search (tsvector)**: Built-in fallback search if Meilisearch is unavailable. The `games.search_vector` column is auto-updated via triggers.
- **ENUM types**: Content ratings, development status, and user roles are enforced at the database level.
- **Mature tooling**: Alembic migrations, asyncpg driver, extensive hosting options.

### Schema Organization

The schema is organized into five domains:

```
Identity     → users, user_profiles
Catalog      → games, game_versions, engines, tags, screenshots
Engagement   → reviews, game_likes, collections, follows
Community    → forum_categories, threads, posts, chat_messages
System       → notifications, reports, audit_logs
```

**Key design decisions:**

- **UUID primary keys**: Avoid sequential ID enumeration; safe for distributed systems and public APIs.
- **Denormalized counters**: `like_count`, `review_count`, `average_score` on the `games` table avoid expensive `COUNT(*)` queries on every page load. Updated asynchronously via Celery.
- **Slug-based URLs**: Every public entity (games, forums, threads) has a unique slug for SEO-friendly URLs.
- **Soft deletes**: User accounts and games are soft-deleted (`is_active=false`) to preserve referential integrity and allow recovery.
- **One review per user per game**: Enforced by a unique constraint on `(game_id, user_id)`.

### Migration Strategy

Alembic auto-generates migrations from SQLAlchemy model changes. Migrations are version-controlled and applied sequentially. Production deployments run `alembic upgrade head` before starting new backend containers.

---

## Caching Strategy

Redis serves multiple roles with a tiered caching approach:

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Client    │────▶│   FastAPI    │────▶│  PostgreSQL  │
└─────────────┘     └──────┬───────┘     └──────────────┘
                           │
                    ┌──────▼───────┐
                    │    Redis     │
                    │              │
                    │ • Hot data   │
                    │ • Sessions   │
                    │ • Rate limit │
                    │ • Pub/sub    │
                    └──────────────┘
```

| Cache target | Key pattern | TTL | Invalidation |
|---|---|---|---|
| Game detail | `game:{slug}` | 5 min | On game update/delete |
| User profile | `user:{username}` | 5 min | On profile update |
| Trending games | `trending:week` | 1 hour | Celery beat recalculates hourly |
| Featured games | `featured` | 30 min | On admin feature/unfeature |
| Similar games | `similar:{game_id}` | 24 hours | Nightly Celery recomputation |
| JWT blocklist | `blocklist:{jti}` | Token remaining lifetime | Automatic (TTL) |
| Rate limit counters | `ratelimit:{ip}:{endpoint}` | Window duration | Automatic (TTL) |

**Cache-aside pattern**: On read, check Redis first. On miss, query PostgreSQL, populate cache, return. On write, update PostgreSQL and invalidate the relevant cache key.

**What is not cached:**

- Search results (Meilisearch handles this with its own in-memory index)
- Real-time chat messages (fetched from DB on channel join, then streamed via WebSocket)
- Write operations (always hit the database directly)

---

## Real-Time Architecture

GameVault uses FastAPI WebSockets with Redis pub/sub for horizontal scalability.

```
┌──────────┐    WS     ┌──────────────┐   pub/sub   ┌──────────────┐
│ Client A │◄─────────▶│  Backend #1  │◄───────────▶│    Redis     │
└──────────┘           └──────────────┘             └──────┬───────┘
                                                          │
┌──────────┐    WS     ┌──────────────┐                    │
│ Client B │◄─────────▶│  Backend #2  │◄───────────────────┘
└──────────┘           └──────────────┘
```

### Chat Flow

1. Client connects to `ws://host/ws/chat/{channel}` with JWT token
2. Backend authenticates, loads last 50 messages from PostgreSQL
3. Client sends a message → backend validates, persists to `chat_messages`, publishes to Redis channel `chat:{channel}`
4. All backend instances subscribed to `chat:{channel}` push the message to their connected WebSocket clients
5. Rate limit: 1 message per 2 seconds per user (Redis counter)

### Notification Stream

1. Client connects to `ws://host/ws/notifications` with JWT token
2. Backend subscribes to Redis channel `notifications:{user_id}`
3. When a Celery task creates a notification, it publishes to the user's Redis channel
4. The backend instance holding the user's WebSocket connection pushes the event

**Why Redis pub/sub over a dedicated message broker (RabbitMQ)?**

Redis is already required for caching, sessions, and rate limiting. Adding pub/sub avoids operating a second message broker. For GameVault's scale (< 10K concurrent WebSocket connections), Redis pub/sub is sufficient.

---

## Search Architecture

Meilisearch provides the user-facing search experience; PostgreSQL `tsvector` serves as a fallback.

```
┌──────────┐         ┌──────────────┐         ┌──────────────┐
│ Frontend │────────▶│   FastAPI    │────────▶│ Meilisearch  │
│          │  /search│  (proxy)     │         │  index:games │
└──────────┘         └──────────────┘         └──────────────┘
                            │
                     ┌──────▼───────┐
                     │   Celery     │
                     │  (index sync)│
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │  PostgreSQL  │
                     └──────────────┘
```

### Index Configuration

- **Index name**: `games`
- **Searchable attributes**: `title`, `synopsis`, `plot`, `author_names`, `engine_name`, `tag_names`
- **Filterable attributes**: `engine_slug`, `development_status`, `rating`, `tag_slugs`, `original_pc_gender`, `has_play_online`, `like_count`
- **Sortable attributes**: `created_at`, `updated_at`, `like_count`, `average_score`, `title`
- **Ranking rules**: `words`, `typo`, `proximity`, `attribute`, `sort`, `exactness`

### Sync Strategy

A Celery task runs on every game create, update, or delete:

1. **Create/Update**: Fetch the full game document from PostgreSQL (with joined author, engine, tags), upsert into Meilisearch
2. **Delete**: Remove the document from Meilisearch by ID

Full re-index available via admin command for disaster recovery.

**Why Meilisearch over Elasticsearch?**

- Lightweight (single binary, < 100 MB RAM for small indexes)
- Typo-tolerant out of the box
- Faceted filtering without complex query DSL
- Sub-100ms search at p95 for < 1M documents
- Simpler operations than an Elasticsearch cluster

**Why not PostgreSQL full-text search alone?**

PostgreSQL `tsvector` lacks typo tolerance, faceted filtering, and relevance tuning. It serves as a reliable fallback but Meilisearch provides a significantly better search UX.

---

## File Upload Strategy

All file uploads use a **presigned URL pattern** to keep binary data off the API server.

```
┌──────────┐  1. presign   ┌──────────┐  2. presigned URL  ┌──────────┐
│  Client  │──────────────▶│ FastAPI  │───────────────────▶│  Client  │
└──────────┘               └──────────┘                    └────┬─────┘
                                                                │
                                                     3. PUT file│
                                                                ▼
                                                           ┌──────────┐
                                                           │  MinIO   │
                                                           └────┬─────┘
                                                                │
┌──────────┐  5. confirm    ┌──────────┐  4. verify exists    │
│  Client  │──────────────▶│ FastAPI  │◄──────────────────────┘
└──────────┘               └────┬─────┘
                                │
                         6. thumbnail
                                ▼
                           ┌──────────┐
                           │  Celery  │
                           └──────────┘
```

### Upload Flow

1. Client requests a presigned URL: `POST /api/v1/uploads/presign`
2. Backend validates file type and size limits, generates a presigned PUT URL from MinIO
3. Client uploads directly to MinIO (bypasses the API server entirely)
4. Client confirms the upload via the relevant endpoint (e.g., `POST /games/{slug}/screenshots`)
5. Backend verifies the object exists in MinIO
6. For images, a Celery task generates thumbnails

### Bucket Organization

| Bucket | Access | Contents |
|---|---|---|
| `game-files` | Private (presigned download URLs) | Game binaries (ZIP, EXE, APK) |
| `screenshots` | Public (via CDN) | Game screenshots and thumbnails |
| `avatars` | Public (via CDN) | User profile avatars |

### Why MinIO over Cloud S3?

- Self-hosted and S3-compatible — no vendor lock-in
- Runs in Docker alongside the rest of the stack for local development
- Same API works in production (swap endpoint to AWS S3 or Cloudflare R2 if preferred)
- No egress costs for self-hosted deployments

### Security

- MIME type validation on presign (whitelist per upload purpose)
- File size limits enforced at presign time
- Presigned URLs expire after 1 hour
- Object keys include a random prefix to prevent enumeration
- Future: ClamAV virus scanning on uploaded game files

---

## Layered Backend Architecture

```
Routes (api/v1/)     → Thin handlers: validate input, call service, return response
    ↓
Services (services/) → Business logic: orchestrate repos, enforce rules, emit events
    ↓
Repositories (repositories/) → Data access: SQLAlchemy queries, no business logic
    ↓
Models (models/)     → SQLAlchemy ORM definitions
```

**Dependency injection** via FastAPI `Depends()`:

- `get_db()` → async database session (auto-commit/rollback)
- `get_current_user()` → JWT validation, user lookup
- `require_role("admin")` → RBAC enforcement
- `get_pagination()` → standardized page/per_page parameters

**Error handling**: Custom exception classes (`NotFoundError`, `ForbiddenError`, `ConflictError`) mapped to consistent JSON error responses via registered exception handlers.

---

## Background Task Architecture

Celery handles all async work that should not block API responses:

| Task | Trigger | Queue |
|---|---|---|
| Search index sync | Game CRUD | `search` |
| Email dispatch | Registration, password reset, notifications | `email` |
| Thumbnail generation | Image upload confirmed | `media` |
| Notification fan-out | Review, reply, follow events | `notifications` |
| Trending score calculation | Celery beat (hourly) | `analytics` |
| Recommendation computation | Celery beat (nightly) | `analytics` |
| Cleanup (expired tokens, old sessions) | Celery beat (daily) | `cleanup` |

Celery beat handles scheduled tasks. Workers are horizontally scalable by adding more `celery-worker` containers.

---

## API Versioning

All routes are prefixed with `/api/v1/`. This allows future breaking changes under `/api/v2/` without disrupting existing clients. The frontend and any third-party API consumers should always target a specific version.

---

## Observability

| Concern | Tool |
|---|---|
| Structured logging | structlog (JSON in production) |
| Metrics | Prometheus `/metrics` endpoint |
| Dashboards | Grafana |
| Distributed tracing | OpenTelemetry → Jaeger (dev) |
| Error tracking | Sentry (backend + frontend) |
| Health checks | `/api/v1/health` (DB, Redis, Meilisearch) |
