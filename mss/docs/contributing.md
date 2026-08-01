# Contributing to GameVault

Thank you for your interest in contributing to GameVault! This guide covers everything you need to get started.

---

## Development Setup

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/your-username/gamevault.git
   cd gamevault
   ```

2. **Copy environment variables**

   ```bash
   cp .env.example .env
   ```

3. **Start the development stack**

   ```bash
   make up
   make migrate
   make seed
   make superuser
   ```

4. **Verify everything works**

   - Frontend: http://localhost
   - API health: http://localhost/api/v1/health
   - API docs: http://localhost/api/v1/docs

Hot reload is enabled by default via `docker-compose.override.yml`. Changes to `backend/app/` and `frontend/src/` are reflected without rebuilding containers.

---

## Code Style

### Python (Backend)

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Inside the backend container
docker compose exec backend ruff check app/
docker compose exec backend ruff format app/
```

**Conventions:**

- Follow PEP 8 (enforced by Ruff)
- Use type hints on all function signatures
- Async functions for all database and I/O operations
- Keep route handlers thin — business logic belongs in `services/`
- Pydantic schemas for all request/response models
- Docstrings on public service methods (Google style)

**Project structure for new features:**

```
app/
├── api/v1/your_domain.py      # Route handlers
├── schemas/your_domain.py     # Pydantic models
├── services/your_domain_service.py  # Business logic
├── repositories/your_domain_repo.py  # Database queries
└── models/your_domain.py      # SQLAlchemy models
```

### TypeScript (Frontend)

We use ESLint and Prettier (via Next.js defaults).

```bash
docker compose exec frontend npm run lint
```

**Conventions:**

- TypeScript strict mode — no `any` unless absolutely necessary
- Functional React components with hooks
- Colocate component-specific types in the same file or in `src/types/`
- Use TanStack Query for server state, Zustand for client state
- Tailwind CSS for styling — no inline styles
- shadcn/ui components in `src/components/ui/`

---

## Git Workflow

### Branching

1. Create a feature branch from `main`:

   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/short-description
   ```

   Branch naming conventions:

   | Prefix | Use for |
   |---|---|
   | `feature/` | New features |
   | `fix/` | Bug fixes |
   | `refactor/` | Code refactoring |
   | `docs/` | Documentation only |
   | `test/` | Test additions or fixes |
   | `chore/` | Tooling, dependencies, CI |

2. Make your changes in small, focused commits.

3. Push and open a pull request:

   ```bash
   git push -u origin feature/short-description
   ```

### Pull Request Process

1. Fill out the PR template with a clear description of changes
2. Link any related issues (`Closes #123`)
3. Ensure CI passes (lint, type-check, tests)
4. Request review from at least one maintainer
5. Address review feedback
6. Squash-merge into `main` after approval

**PR guidelines:**

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Update documentation if you change APIs or configuration
- Add screenshots for UI changes

---

## Testing Requirements

### Backend

```bash
make test-backend
```

**Requirements:**

- All existing tests must pass
- New API endpoints require integration tests
- New service methods require unit tests
- Target: 80%+ code coverage on new code

**Test structure:**

```
tests/
├── conftest.py          # Shared fixtures
├── unit/                # Service and utility tests
├── integration/         # API endpoint tests (httpx TestClient)
└── e2e/                 # Full flow tests
```

**Example integration test:**

```python
async def test_create_game(client, auth_headers):
    response = await client.post(
        "/api/v1/games",
        json={"title": "Test Game", "engine_id": 1},
        headers=auth_headers,
    )
    assert response.status_code == 201
    assert response.json()["title"] == "Test Game"
```

### Frontend

```bash
make test-frontend
```

**Requirements:**

- Component tests for critical UI (forms, auth flows)
- Use Vitest + Testing Library
- E2E tests with Playwright for core user flows (register, submit game, search)

---

## Commit Message Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `chore` | Build process, dependencies, tooling |
| `perf` | Performance improvement |

### Scope

Use the domain area: `auth`, `games`, `reviews`, `forums`, `chat`, `search`, `admin`, `frontend`, `infra`.

### Examples

```
feat(games): add version history endpoint

fix(auth): prevent token refresh race condition

docs(api): document search query parameters

test(reviews): add integration tests for review voting

chore(deps): update fastapi to 0.115.0
```

### Rules

- Use imperative mood ("add" not "added" or "adds")
- First line ≤ 72 characters
- Reference issues in the footer: `Closes #42`
- Breaking changes: `feat(api)!: rename /games/list to /games`

---

## Database Migrations

When modifying SQLAlchemy models:

```bash
make makemigrations msg="add game_follows table"
make migrate
```

**Migration guidelines:**

- Review auto-generated migrations before committing
- Never edit a migration that has been merged to `main`
- Include both `upgrade()` and `downgrade()` functions
- Test migrations on a fresh database: `make reset-db`

---

## Reporting Issues

- Use GitHub Issues with the appropriate template
- Include steps to reproduce, expected vs. actual behavior
- Attach screenshots for UI bugs
- Include relevant logs (`make logs`)

---

## Code of Conduct

Be respectful and constructive. GameVault is a community project — treat all contributors with kindness. Harassment, discrimination, and toxic behavior are not tolerated.

---

## Questions?

Open a [GitHub Discussion](https://github.com/your-org/gamevault/discussions) or reach out to the maintainers. We're happy to help you get started!
