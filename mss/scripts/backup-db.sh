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
BACKUP_DIR="${BACKUP_DIR:-./backups}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_FILE="${BACKUP_DIR}/gamevault_${TIMESTAMP}.sql"

mkdir -p "${BACKUP_DIR}"

echo "Backing up database '${DB_NAME}' to ${BACKUP_FILE}..."
docker compose exec -T postgres pg_dump -U "${DB_USER}" "${DB_NAME}" > "${BACKUP_FILE}"

echo "Backup complete: ${BACKUP_FILE}"
