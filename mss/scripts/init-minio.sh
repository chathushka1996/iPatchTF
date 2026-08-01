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

MINIO_ENDPOINT="${MINIO_ENDPOINT:-minio:9000}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"
MINIO_BUCKET_GAMES="${MINIO_BUCKET_GAMES:-game-files}"
MINIO_BUCKET_SCREENSHOTS="${MINIO_BUCKET_SCREENSHOTS:-screenshots}"
MINIO_BUCKET_AVATARS="${MINIO_BUCKET_AVATARS:-avatars}"

echo "Waiting for MinIO at http://${MINIO_ENDPOINT}..."
until docker run --rm --network gamevault curlimages/curl:8.5.0 -sf "http://${MINIO_ENDPOINT}/minio/health/live" > /dev/null 2>&1; do
  sleep 2
done

echo "Creating MinIO buckets..."
docker run --rm --network gamevault minio/mc:latest sh -c "
  mc alias set local http://${MINIO_ENDPOINT} ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}
  mc mb --ignore-existing local/${MINIO_BUCKET_GAMES}
  mc mb --ignore-existing local/${MINIO_BUCKET_SCREENSHOTS}
  mc mb --ignore-existing local/${MINIO_BUCKET_AVATARS}
  mc anonymous set download local/${MINIO_BUCKET_SCREENSHOTS}
"

echo "MinIO buckets ready: ${MINIO_BUCKET_GAMES}, ${MINIO_BUCKET_SCREENSHOTS}, ${MINIO_BUCKET_AVATARS}"
