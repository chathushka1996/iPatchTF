# GameVault

**Discover, share, and discuss interactive games — a community-driven game database and social platform.**

GameVault is a full-stack platform for browsing, submitting, reviewing, and discussing games. It combines a rich game catalog with forums, real-time chat, collections, and a modern web UI — all runnable locally with a single `make up` command.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Next.js 14 (App Router) | SSR, SEO-friendly game pages, React Server Components |
| Backend API | FastAPI (Python 3.12+) | Async REST API, WebSockets, auto-generated OpenAPI docs |
| Database | PostgreSQL 16 | Primary relational data store |
| Cache / Broker | Redis 7 | Sessions, rate limiting, pub/sub, Celery broker |
| Search | Meilisearch | Typo-tolerant instant search with faceted filtering |
| Object Storage | MinIO | S3-compatible storage for game files, screenshots, avatars |
| Background Tasks | Celery | Email, search index sync, notifications, cleanup |
| Reverse Proxy | Traefik v3 | Routing, TLS termination, service discovery |
| Real-time | FastAPI WebSockets + Redis pub/sub | Live chat and notification streams |

---

## Architecture

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

All traffic enters through Traefik. API and WebSocket routes (`/api/*`, `/ws/*`) are forwarded to FastAPI; all other paths go to Next.js. The backend persists data in PostgreSQL, caches in Redis, indexes search in Meilisearch, and stores files in MinIO. Celery workers handle async tasks via Redis.

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose v2
- [Make](https://www.gnu.org/software/make/) (optional but recommended)
- Git

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/gamevault.git
   cd gamevault
   ```

2. **Copy environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and change `SECRET_KEY`, database passwords, and other secrets before deploying.

3. **Start all services**

   ```bash
   make up
   ```

   This builds and starts Traefik, backend, frontend, PostgreSQL, Redis, Meilisearch, MinIO, and Celery workers.

4. **Run database migrations**

   ```bash
   make migrate
   ```

5. **Seed sample data**

   ```bash
   make seed
   ```

6. **Create an admin account**

   ```bash
   make superuser
   ```

7. **Open the app**

   Visit [http://localhost](http://localhost)

   | Service | URL |
   |---|---|
   | Frontend | http://localhost |
   | API | http://localhost/api/v1 |
   | API docs (Swagger) | http://localhost/api/v1/docs |
   | Traefik dashboard | http://localhost:8080 |
   | MinIO console | http://localhost:9001 |

---

## Development

### Hot Reload

Development overrides in `docker-compose.override.yml` mount source directories into containers:

- **Backend**: `./backend/app` → `/app/app` (uvicorn `--reload`)
- **Frontend**: `./frontend/src` → `/app/src` (Next.js dev server)

Changes to Python or TypeScript source files are picked up automatically without rebuilding images.

### Running Tests

```bash
# Backend (pytest)
make test-backend

# Frontend (Jest)
make test-frontend
```

### Making Migrations

After changing SQLAlchemy models:

```bash
make makemigrations msg="describe your change"
make migrate
```

### Interactive Shell

```bash
make shell          # Python REPL inside the backend container
make logs           # Tail logs from all services
make reset-db       # Wipe volumes, restart, migrate, and seed
```

---

## Make Commands

| Command | Description |
|---|---|
| `make up` | Build and start all services in the background |
| `make down` | Stop and remove containers |
| `make logs` | Follow logs from all services |
| `make migrate` | Apply Alembic migrations (`upgrade head`) |
| `make makemigrations msg="..."` | Generate a new Alembic migration |
| `make seed` | Populate the database with sample data |
| `make superuser` | Create an admin user interactively |
| `make test-backend` | Run backend tests with pytest |
| `make test-frontend` | Run frontend tests with Jest |
| `make shell` | Open a Python shell in the backend container |
| `make reset-db` | Destroy volumes, restart, migrate, and seed |

---

## Project Structure

```
gamevault/
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── api/v1/           # Route handlers (auth, games, forums, …)
│   │   ├── models/           # SQLAlchemy ORM models
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   ├── services/         # Business logic layer
│   │   ├── repositories/     # Data access layer
│   │   ├── middleware/       # CORS, rate limiting, request ID
│   │   └── tasks/            # Celery task definitions
│   ├── alembic/              # Database migrations
│   ├── scripts/              # seed_db.py, create_superuser.py
│   └── tests/
├── frontend/                 # Next.js application
│   └── src/
│       ├── app/              # App Router pages
│       ├── components/       # UI components
│       ├── hooks/            # Custom React hooks
│       ├── lib/              # API client, auth, utilities
│       └── stores/           # Zustand state stores
├── docs/                     # Documentation
│   ├── api.md
│   ├── architecture.md
│   ├── contributing.md
│   └── deployment.md
├── scripts/                  # init-db.sh, init-minio.sh, backup-db.sh
├── traefik/                  # Traefik configuration
├── docker-compose.yml
├── docker-compose.override.yml
├── Makefile
└── .env.example
```

---

## API Documentation

Interactive OpenAPI documentation is available at:

**http://localhost/api/v1/docs**

A static endpoint reference is maintained in [docs/api.md](docs/api.md).

---

## Contributing

We welcome contributions! Please read [docs/contributing.md](docs/contributing.md) for:

- Development setup and workflow
- Code style guidelines (Ruff, ESLint, Prettier)
- Git branching and pull request process
- Testing requirements
- Commit message conventions

---

## Documentation

| Document | Description |
|---|---|
| [docs/api.md](docs/api.md) | Full API endpoint reference |
| [docs/architecture.md](docs/architecture.md) | Architecture decisions and design rationale |
| [docs/deployment.md](docs/deployment.md) | Production deployment guide |
| [docs/contributing.md](docs/contributing.md) | Contributor guidelines |

---

## License

This project is licensed under the [MIT License](LICENSE).
