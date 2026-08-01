#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

DB_USER="${DB_USER:-gamevault}"
DB_NAME="${DB_NAME:-gamevault}"

echo "Waiting for PostgreSQL..."
until docker compose exec -T postgres pg_isready -U "${DB_USER}" -d "${DB_NAME}" > /dev/null 2>&1; do
  sleep 2
done

echo "Running Alembic migrations..."
docker compose exec -T backend alembic upgrade head

echo "Seeding database..."
docker compose exec -T backend python scripts/seed_db.py

echo "Database initialization complete."
