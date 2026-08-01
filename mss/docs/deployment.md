# GameVault Production Deployment Guide

This guide covers deploying GameVault to a production environment with TLS, managed services, monitoring, and backups.

---

## Prerequisites

- A Linux server (or container orchestration platform) with Docker and Docker Compose
- A domain name pointed at your server
- SMTP credentials for transactional email
- (Recommended) Managed PostgreSQL, or a dedicated database server

---

## Environment Variables Checklist

Copy `.env.example` to `.env` and configure every variable for production.

### Application

| Variable | Required | Notes |
|---|---|---|
| `APP_NAME` | Yes | `GameVault` |
| `APP_ENV` | Yes | Set to `production` |
| `SECRET_KEY` | Yes | 64+ character random hex string |
| `ALLOWED_ORIGINS` | Yes | `https://yourdomain.com` (no trailing slash) |

### Database

| Variable | Required | Notes |
|---|---|---|
| `DB_HOST` | Yes | Managed DB hostname or internal service name |
| `DB_PORT` | Yes | Usually `5432` |
| `DB_NAME` | Yes | Database name |
| `DB_USER` | Yes | Database user with least privilege |
| `DB_PASSWORD` | Yes | Strong, unique password |
| `DATABASE_URL` | Yes | Full async connection string |

### Redis

| Variable | Required | Notes |
|---|---|---|
| `REDIS_URL` | Yes | `redis://host:6379/0` |
| `CELERY_BROKER_URL` | Yes | `redis://host:6379/1` (separate DB index) |

### Meilisearch

| Variable | Required | Notes |
|---|---|---|
| `MEILI_URL` | Yes | Internal URL, e.g. `http://meilisearch:7700` |
| `MEILI_MASTER_KEY` | Yes | Strong random key (not the dev default) |
| `MEILI_ENV` | Yes | Set to `production` |

### MinIO / Object Storage

| Variable | Required | Notes |
|---|---|---|
| `MINIO_ENDPOINT` | Yes | Internal endpoint |
| `MINIO_ACCESS_KEY` | Yes | Strong access key |
| `MINIO_SECRET_KEY` | Yes | Strong secret key |
| `MINIO_BUCKET_GAMES` | Yes | `game-files` |
| `MINIO_BUCKET_SCREENSHOTS` | Yes | `screenshots` |
| `MINIO_BUCKET_AVATARS` | Yes | `avatars` |

### Auth

| Variable | Required | Notes |
|---|---|---|
| `JWT_ALGORITHM` | Yes | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Yes | `30` recommended |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Yes | `7` recommended |

### Email

| Variable | Required | Notes |
|---|---|---|
| `SMTP_HOST` | Yes | Your SMTP provider hostname |
| `SMTP_PORT` | Yes | Usually `587` (TLS) |
| `SMTP_USER` | Yes | SMTP username |
| `SMTP_PASSWORD` | Yes | SMTP password or app token |
| `EMAIL_FROM` | Yes | `GameVault <noreply@yourdomain.com>` |

### Frontend

| Variable | Required | Notes |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Yes | `https://yourdomain.com/api` |
| `NEXT_PUBLIC_WS_URL` | Yes | `wss://yourdomain.com/ws` |
| `NEXT_PUBLIC_MINIO_PUBLIC_URL` | Yes | CDN URL for public assets |

---

## Docker Production Build

Use production Dockerfile targets instead of development:

```yaml
# docker-compose.prod.yml (excerpt)
services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
      target: production
    restart: unless-stopped
    env_file: .env
    # No volume mounts in production

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      target: production
    restart: unless-stopped
    env_file: .env

  celery-worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
      target: production
    command: celery -A app.tasks worker --loglevel=warning --concurrency=8

  celery-beat:
    build:
      context: ./backend
      dockerfile: Dockerfile
      target: production
    command: celery -A app.tasks beat --loglevel=warning
```

Build and deploy:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml build
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
docker compose exec backend alembic upgrade head
```

---

## Traefik TLS with Let's Encrypt

Update `traefik/traefik.yml` for production:

```yaml
api:
  dashboard: false

entryPoints:
  web:
    address: ":80"
    http:
      redirections:
        entryPoint:
          to: websecure
          scheme: https
  websecure:
    address: ":443"

certificatesResolvers:
  letsencrypt:
    acme:
      email: admin@yourdomain.com
      storage: /letsencrypt/acme.json
      httpChallenge:
        entryPoint: web

providers:
  docker:
    endpoint: "unix:///var/run/docker.sock"
    exposedByDefault: false
    network: gamevault

log:
  level: WARN
```

Add Traefik labels to services:

```yaml
labels:
  - "traefik.enable=true"
  - "traefik.http.routers.api.rule=Host(`yourdomain.com`) && (PathPrefix(`/api`) || PathPrefix(`/ws`))"
  - "traefik.http.routers.api.entrypoints=websecure"
  - "traefik.http.routers.api.tls.certresolver=letsencrypt"
  - "traefik.http.services.api.loadbalancer.server.port=8000"
```

Mount the Let's Encrypt storage volume:

```yaml
traefik:
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock:ro
    - ./traefik/traefik.yml:/etc/traefik/traefik.yml:ro
    - letsencrypt_data:/letsencrypt
```

---

## Database Setup (Managed PostgreSQL)

For production, use a managed PostgreSQL service (AWS RDS, DigitalOcean Managed Databases, Supabase, etc.).

### Recommendations

- **PostgreSQL 16** to match the development environment
- Enable **SSL/TLS** connections (`sslmode=require` in connection string)
- Set `max_connections` appropriately (use PgBouncer for connection pooling if needed)
- Enable automated backups at the provider level
- Use a dedicated database user with minimal required permissions

### Connection String

```
DATABASE_URL=postgresql+asyncpg://gamevault:PASSWORD@db-host.example.com:5432/gamevault?ssl=require
```

### Migrations

Run migrations as a one-off task before starting new backend containers:

```bash
docker compose exec backend alembic upgrade head
```

---

## MinIO Behind a CDN

Serve public assets (screenshots, avatars) through a CDN for lower latency and reduced origin load.

### Setup

1. Configure MinIO buckets with a public-read policy for `screenshots` and `avatars`
2. Point a CDN origin at your MinIO public endpoint (or use CloudFront, Cloudflare R2, etc.)
3. Set `NEXT_PUBLIC_MINIO_PUBLIC_URL` to the CDN URL:

   ```
   NEXT_PUBLIC_MINIO_PUBLIC_URL=https://cdn.yourdomain.com
   ```

4. Keep `game-files` bucket private — downloads use presigned URLs generated by the API

### CORS

Configure MinIO CORS to allow uploads from your frontend domain:

```json
[
  {
    "AllowedOrigins": ["https://yourdomain.com"],
    "AllowedMethods": ["GET", "PUT"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

---

## Monitoring (Prometheus + Grafana)

### Prometheus

Expose metrics from the FastAPI backend at `/metrics`. Add a Prometheus scrape config:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: gamevault-backend
    static_configs:
      - targets: ['backend:8000']
    metrics_path: /metrics
    scrape_interval: 15s

  - job_name: postgres
  # Use postgres_exporter sidecar

  - job_name: redis
  # Use redis_exporter sidecar
```

Key metrics to monitor:

| Metric | Alert threshold |
|---|---|
| HTTP request rate | Baseline + 3σ |
| HTTP 5xx error rate | > 1% over 5 min |
| API p95 latency | > 500ms |
| Database connection pool usage | > 80% |
| Redis memory usage | > 80% of maxmemory |
| Celery queue depth | > 1000 tasks |
| Celery task failure rate | > 5% over 15 min |

### Grafana

Import or create dashboards for:

- **API Overview**: request rate, latency histogram, error rate, active connections
- **Database**: query time, connection count, table sizes
- **Redis**: memory, hit rate, connected clients
- **Celery**: queue depth, task duration, worker count
- **Infrastructure**: CPU, memory, disk I/O per container

### Health Checks

Configure your orchestrator to probe `/api/v1/health`. The endpoint returns `degraded` if any dependency (database, Redis, Meilisearch) is unhealthy.

### Error Tracking

Integrate [Sentry](https://sentry.io) for both backend and frontend to capture exceptions with stack traces and request context.

---

## Backup Strategy

### PostgreSQL

**Daily full backups** via `pg_dump`:

```bash
# scripts/backup-db.sh (already in repo)
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME | gzip > backup-$(date +%Y%m%d).sql.gz
```

Schedule with cron or a Celery beat task. Store backups in MinIO or off-site object storage. Retain 30 daily backups, 12 monthly.

For managed PostgreSQL, enable the provider's automated backup feature as the primary strategy and use `pg_dump` as a secondary export.

**Point-in-time recovery:** Enable WAL archiving for managed databases.

### MinIO

- Enable **versioning** on all buckets
- Schedule `mc mirror` to replicate to a secondary storage location
- For production at scale, use MinIO distributed mode or migrate to S3 with cross-region replication

### Redis

- AOF persistence is enabled in `docker-compose.yml` (`appendonly yes`)
- Schedule periodic RDB snapshots
- Redis data is ephemeral (cache, sessions) — plan for cold-cache recovery after restore

### Meilisearch

Schedule daily snapshots via the Meilisearch API:

```bash
curl -X POST "http://meilisearch:7700/snapshots" \
  -H "Authorization: Bearer $MEILI_MASTER_KEY"
```

Re-index from PostgreSQL if snapshots are lost (Celery task can rebuild the index).

---

## Scaling Considerations

### Horizontal Scaling

| Component | Strategy |
|---|---|
| **Backend** | Stateless — add more containers behind Traefik load balancer |
| **Frontend** | Stateless — add more Next.js containers |
| **Celery workers** | Add worker containers; increase `--concurrency` per worker |
| **PostgreSQL** | Vertical scaling first; read replicas for browse/search queries (future) |
| **Redis** | Redis Sentinel or Cluster for HA; separate instances for cache vs. broker |
| **Meilisearch** | Single node handles < 1M documents; shard if needed |
| **MinIO** | Distributed mode across 4+ nodes for HA |

### Connection Pooling

Use PgBouncer in front of PostgreSQL when running multiple backend workers:

```
backend (4 workers) × N instances = many connections
PgBouncer → pools to ~20 actual DB connections
```

### Caching

- Redis caches hot game detail pages (TTL: 5 min)
- CDN caches static assets and SSR pages (configure `Cache-Control` headers)
- Meilisearch handles search — no need to cache search results in Redis

### WebSocket Scaling

WebSocket connections are sticky to a backend instance. Use Redis pub/sub (already implemented) so messages reach clients on any instance. Configure Traefik sticky sessions or use a dedicated WebSocket service.

### Resource Guidelines (starting point)

| Service | CPU | Memory |
|---|---|---|
| Backend (per instance) | 1 vCPU | 512 MB |
| Frontend (per instance) | 0.5 vCPU | 256 MB |
| Celery worker | 1 vCPU | 512 MB |
| PostgreSQL | 2 vCPU | 2 GB |
| Redis | 0.5 vCPU | 512 MB |
| Meilisearch | 1 vCPU | 1 GB |
| MinIO | 1 vCPU | 1 GB |

Scale based on observed metrics from Prometheus/Grafana.

---

## Security Checklist

- [ ] `APP_ENV=production` — disables debug modes
- [ ] Strong, unique `SECRET_KEY` and `MEILI_MASTER_KEY`
- [ ] TLS enabled on all public endpoints
- [ ] CORS restricted to production domain
- [ ] Database connections use SSL
- [ ] MinIO credentials rotated from defaults
- [ ] Traefik dashboard disabled or protected
- [ ] Rate limiting enabled (Redis-backed)
- [ ] Log rotation configured
- [ ] Secrets stored in environment variables or a secrets manager (not in git)
- [ ] Regular dependency updates (Dependabot/Snyk)

---

## Post-Deploy Verification

```bash
# Health check
curl https://yourdomain.com/api/v1/health

# TLS certificate
curl -vI https://yourdomain.com 2>&1 | grep "SSL certificate"

# Search
curl "https://yourdomain.com/api/v1/search?q=test"

# Frontend
curl -o /dev/null -s -w "%{http_code}" https://yourdomain.com
```

All checks should return `200`. Review logs for startup errors:

```bash
docker compose logs backend --tail=50
docker compose logs celery-worker --tail=50
```
