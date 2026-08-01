# GameVault API Reference

Base URL: `http://localhost/api/v1`

All authenticated endpoints require a Bearer token in the `Authorization` header, or an HTTP-only cookie set at login.

**Response types:**

| Type | Description |
|---|---|
| `JSON` | Standard JSON response body |
| `Paginated` | `{ items: [...], total, page, per_page, pages }` |
| `204` | No content (successful delete/update) |
| `WebSocket` | Persistent bidirectional connection |

**Error format:**

```json
{
  "detail": "Human-readable error message",
  "error_code": "UNIQUE_CODE",
  "status": 400
}
```

---

## Health

| Method | Path | Description | Auth | Response |
|---|---|---|---|---|
| `GET` | `/health` | Service health and dependency status | No | `JSON` |

### Example: Health Check

**Request:**

```http
GET /api/v1/health
```

**Response `200`:**

```json
{
  "status": "ok",
  "service": "GameVault API",
  "version": "1.0.0",
  "dependencies": {
    "database": { "status": "ok" },
    "redis": { "status": "ok" },
    "meilisearch": { "status": "ok" }
  }
}
```

---

## Auth

| Method | Path | Description | Auth | Response |
|---|---|---|---|---|
| `POST` | `/auth/register` | Create a new account | No | `JSON` (user + tokens) |
| `POST` | `/auth/login` | Authenticate and receive tokens | No | `JSON` (tokens) |
| `POST` | `/auth/refresh` | Exchange refresh token for new access token | Refresh token | `JSON` (tokens) |
| `POST` | `/auth/logout` | Blacklist tokens in Redis | Yes | `204` |
| `POST` | `/auth/forgot-password` | Send password reset email | No | `JSON` |
| `POST` | `/auth/reset-password` | Consume reset token and set new password | No | `JSON` |
| `POST` | `/auth/verify-email/{token}` | Verify email address | No | `JSON` |
| `POST` | `/auth/2fa/setup` | Generate TOTP secret and QR code | Yes | `JSON` |
| `POST` | `/auth/2fa/verify` | Confirm TOTP setup | Yes | `JSON` |
| `POST` | `/auth/2fa/disable` | Disable two-factor authentication | Yes | `JSON` |
| `GET` | `/auth/oauth/{provider}` | Initiate OAuth2 redirect (Google, GitHub, Discord) | No | Redirect |
| `GET` | `/auth/oauth/{provider}/callback` | OAuth2 callback handler | No | `JSON` (tokens) |

### Example: Register

**Request:**

```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "gamedev42",
  "email": "dev@example.com",
  "password": "SecurePass123!"
}
```

**Response `201`:**

```json
{
  "user": {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "username": "gamedev42",
    "email": "dev@example.com",
    "display_name": null,
    "role": "user",
    "is_verified": false,
    "created_at": "2026-07-31T12:00:00Z"
  },
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

### Example: Login

**Request:**

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "dev@example.com",
  "password": "SecurePass123!"
}
```

**Response `200`:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

---

## Users

| Method | Path | Description | Auth | Response |
|---|---|---|---|---|
| `GET` | `/users/me` | Get current user profile | Yes | `JSON` |
| `PUT` | `/users/me` | Update current user profile | Yes | `JSON` |
| `DELETE` | `/users/me` | Soft-delete account | Yes | `204` |
| `GET` | `/users/{username}` | Get public profile | No | `JSON` |
| `GET` | `/users/{username}/games` | List games by user | No | `Paginated` |
| `GET` | `/users/{username}/reviews` | List reviews by user | No | `Paginated` |
| `GET` | `/users/{username}/collections` | List public collections | No | `Paginated` |
| `POST` | `/users/{username}/follow` | Follow a user | Yes | `JSON` |
| `DELETE` | `/users/{username}/follow` | Unfollow a user | Yes | `204` |
| `GET` | `/users/{username}/followers` | List followers | No | `Paginated` |
| `GET` | `/users/{username}/following` | List following | No | `Paginated` |
| `PUT` | `/users/me/avatar` | Upload avatar (multipart) | Yes | `JSON` |
| `PUT` | `/users/me/password` | Change password | Yes | `JSON` |
| `PUT` | `/users/me/notifications` | Update notification preferences | Yes | `JSON` |

### Example: Get Current User

**Request:**

```http
GET /api/v1/users/me
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

**Response `200`:**

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "username": "gamedev42",
  "email": "dev@example.com",
  "display_name": "Game Dev",
  "avatar_url": "https://cdn.gamevault.dev/avatars/a1b2c3d4.png",
  "bio": "Indie game developer",
  "role": "user",
  "is_verified": true,
  "created_at": "2026-07-31T12:00:00Z"
}
```

---

## Games

| Method | Path | Description | Auth | Response |
|---|---|---|---|---|
| `GET` | `/games` | List/search games with filters | No | `Paginated` |
| `GET` | `/games/{slug}` | Get game detail | No | `JSON` |
| `POST` | `/games` | Submit a new game | Yes | `JSON` |
| `PUT` | `/games/{slug}` | Update game metadata | Yes (author/admin) | `JSON` |
| `DELETE` | `/games/{slug}` | Soft-delete game | Admin | `204` |
| `POST` | `/games/{slug}/versions` | Add a new version | Yes (author) | `JSON` |
| `GET` | `/games/{slug}/versions` | List version history | No | `JSON` |
| `PUT` | `/games/{slug}/versions/{id}` | Edit version info | Yes (author) | `JSON` |
| `POST` | `/games/{slug}/screenshots` | Confirm screenshot upload | Yes (author) | `JSON` |
| `DELETE` | `/games/{slug}/screenshots/{id}` | Delete screenshot | Yes (author) | `204` |
| `POST` | `/games/{slug}/like` | Toggle like | Yes | `JSON` |
| `GET` | `/games/{slug}/similar` | Get similar games | No | `JSON` |
| `POST` | `/games/{slug}/follow` | Follow game for updates | Yes | `JSON` |
| `DELETE` | `/games/{slug}/follow` | Unfollow game | Yes | `204` |
| `GET` | `/games/featured` | Featured games | No | `Paginated` |
| `GET` | `/games/trending` | Trending this week | No | `Paginated` |
| `GET` | `/games/recent` | Recently submitted | No | `Paginated` |
| `GET` | `/games/recently-updated` | Recently updated | No | `Paginated` |

**Query parameters for `GET /games`:**

| Parameter | Type | Description |
|---|---|---|
| `q` | string | Full-text search query |
| `engine` | string[] | Filter by engine slug(s) |
| `status` | string[] | Development status |
| `genre` | string[] | Genre tag slugs |
| `adult_theme` | string[] | Adult theme tag slugs |
| `transformation` | string[] | Transformation theme slugs |
| `multimedia` | string[] | Multimedia type slugs |
| `content_warning` | string[] | Content warning slugs |
| `rating` | string[] | Content rating (G, PG, R, X, XXX) |
| `pc_gender` | string[] | Original PC gender |
| `author` | string | Author username |
| `has_play_online` | boolean | Filter play-online games |
| `min_likes` | int | Minimum like count |
| `sort` | string | `newest`, `updated`, `rating`, `likes`, `title`, `trending` |
| `page` | int | Page number (default 1) |
| `per_page` | int | Items per page (default 24, max 100) |

### Example: Get Game Detail

**Request:**

```http
GET /api/v1/games/my-awesome-game
```

**Response `200`:**

```json
{
  "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "title": "My Awesome Game",
  "slug": "my-awesome-game",
  "synopsis": "An epic adventure awaits...",
  "engine": { "id": 1, "name": "Ren'Py", "slug": "renpy" },
  "author": { "username": "gamedev42", "display_name": "Game Dev" },
  "development_status": "beta",
  "rating": "PG",
  "like_count": 142,
  "review_count": 28,
  "average_score": 8.5,
  "tags": [
    { "name": "Adventure", "slug": "adventure", "category": "genre" }
  ],
  "screenshots": [
    { "id": "...", "image_url": "https://...", "thumbnail_url": "https://..." }
  ],
  "versions": [
    {
      "id": "...",
      "version_string": "0.3.0",
      "release_date": "2026-07-15",
      "is_latest": true,
      "downloads": [
        { "label": "Direct", "url": "https://...", "platform": "windows" }
      ]
    }
  ],
  "created_at": "2026-01-15T10:00:00Z",
  "updated_at": "2026-07-15T14:30:00Z"
}
```

### Example: Submit Game

**Request:**

```http
POST /api/v1/games
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
Content-Type: application/json

{
  "title": "My New Game",
  "engine_id": 1,
  "synopsis": "A short description of the game.",
  "development_status": "alpha",
  "rating": "PG",
  "tag_ids": [1, 5, 12]
}
```

**Response `201`:**

```json
{
  "id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
  "title": "My New Game",
  "slug": "my-new-game",
  "is_approved": true,
  "created_at": "2026-07-31T15:00:00Z"
}
```

---

## Reviews

| Method | Path | Description | Auth | Response |
|---|---|---|---|---|
| `GET` | `/games/{slug}/reviews` | List reviews for a game | No | `Paginated` |
| `POST` | `/games/{slug}/reviews` | Submit a review | Yes | `JSON` |
| `PUT` | `/reviews/{id}` | Edit own review | Yes | `JSON` |
| `DELETE` | `/reviews/{id}` | Delete own review | Yes | `204` |
| `POST` | `/reviews/{id}/vote` | Vote helpful/not helpful | Yes | `JSON` |
| `GET` | `/reviews/recent` | Global recent reviews feed | No | `Paginated` |

### Example: Submit Review

**Request:**

```http
POST /api/v1/games/my-awesome-game/reviews
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
Content-Type: application/json

{
  "score": 9,
  "body": "Fantastic story and characters. Highly recommended!",
  "version_reviewed": "0.3.0"
}
```

**Response `201`:**

```json
{
  "id": "d4e5f6a7-b8c9-0123-def0-234567890123",
  "game_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "user": { "username": "player99", "display_name": "Player 99" },
  "score": 9,
  "body": "Fantastic story and characters. Highly recommended!",
  "version_reviewed": "0.3.0",
  "helpful_count": 0,
  "created_at": "2026-07-31T16:00:00Z"
}
```

---

## Forums

| Method | Path | Description | Auth | Response |
|---|---|---|---|---|
| `GET` | `/forums` | List forum categories | No | `JSON` |
| `GET` | `/forums/{slug}` | List threads in a category | No | `Paginated` |
| `POST` | `/forums/{slug}/threads` | Create a thread | Yes | `JSON` |
| `GET` | `/threads/{slug}` | Get thread with posts | No | `JSON` |
| `POST` | `/threads/{slug}/posts` | Reply to a thread | Yes | `JSON` |
| `PUT` | `/posts/{id}` | Edit own post | Yes | `JSON` |
| `DELETE` | `/posts/{id}` | Delete own post | Yes | `204` |
| `POST` | `/threads/{slug}/lock` | Lock thread | Moderator | `JSON` |
| `POST` | `/threads/{slug}/pin` | Pin thread | Moderator | `JSON` |

### Example: Create Thread

**Request:**

```http
POST /api/v1/forums/general-discussion/threads
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
Content-Type: application/json

{
  "title": "What's your favorite engine?",
  "body": "I've been using Ren'Py lately. What about you?"
}
```

**Response `201`:**

```json
{
  "id": "e5f6a7b8-c9d0-1234-ef01-345678901234",
  "title": "What's your favorite engine?",
  "slug": "whats-your-favorite-engine",
  "forum_category": { "name": "General Discussion", "slug": "general-discussion" },
  "author": { "username": "gamedev42" },
  "post_count": 1,
  "created_at": "2026-07-31T17:00:00Z"
}
```

---

## Chat

| Method | Path | Description | Auth | Response |
|---|---|---|---|---|
| `WS` | `/ws/chat/{channel}` | Join a chat channel | Yes | `WebSocket` |
| `WS` | `/ws/notifications` | Real-time notification stream | Yes | `WebSocket` |

**Channels:** `general` (global), `game:{game_id}` (per-game).

Messages are persisted and fanned out via Redis pub/sub. Rate limit: 1 message per 2 seconds per user.

### Example: WebSocket Chat Message

**Connect:**

```
ws://localhost/ws/chat/general?token=eyJhbGciOiJIUzI1NiIs...
```

**Send:**

```json
{
  "type": "message",
  "body": "Hello everyone!"
}
```

**Receive:**

```json
{
  "type": "message",
  "id": "f6a7b8c9-d0e1-2345-f012-456789012345",
  "channel": "general",
  "user": { "username": "gamedev42", "avatar_url": "https://..." },
  "body": "Hello everyone!",
  "created_at": "2026-07-31T18:00:00Z"
}
```

---

## Notifications

| Method | Path | Description | Auth | Response |
|---|---|---|---|---|
| `GET` | `/notifications` | Paginated notification list | Yes | `Paginated` |
| `GET` | `/notifications/unread-count` | Unread notification count | Yes | `JSON` |
| `POST` | `/notifications/mark-read` | Mark specific or all as read | Yes | `JSON` |
| `DELETE` | `/notifications/{id}` | Delete a notification | Yes | `204` |

### Example: Unread Count

**Request:**

```http
GET /api/v1/notifications/unread-count
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

**Response `200`:**

```json
{
  "count": 5
}
```

---

## Collections

| Method | Path | Description | Auth | Response |
|---|---|---|---|---|
| `GET` | `/collections` | Public collections feed | No | `Paginated` |
| `POST` | `/collections` | Create a collection | Yes | `JSON` |
| `GET` | `/collections/{id}` | Collection detail with games | No | `JSON` |
| `PUT` | `/collections/{id}` | Update collection | Yes (owner) | `JSON` |
| `DELETE` | `/collections/{id}` | Delete collection | Yes (owner) | `204` |
| `POST` | `/collections/{id}/games` | Add game to collection | Yes (owner) | `JSON` |
| `DELETE` | `/collections/{id}/games/{game_id}` | Remove game from collection | Yes (owner) | `204` |
| `PUT` | `/collections/{id}/games/reorder` | Reorder games in collection | Yes (owner) | `JSON` |

---

## Search

| Method | Path | Description | Auth | Response |
|---|---|---|---|---|
| `GET` | `/search` | Full-text search (proxied to Meilisearch) | No | `JSON` |
| `GET` | `/search/suggestions` | Autocomplete suggestions | No | `JSON` |

**Query parameters for `GET /search`:**

| Parameter | Type | Description |
|---|---|---|
| `q` | string | Search query (required) |
| `engine` | string[] | Filter by engine slug |
| `status` | string[] | Development status |
| `rating` | string[] | Content rating |
| `tag` | string[] | Tag slugs |
| `sort` | string | Sort field |
| `page` | int | Page number |
| `per_page` | int | Results per page |

### Example: Search

**Request:**

```http
GET /api/v1/search?q=adventure&engine=renpy&sort=likes&per_page=10
```

**Response `200`:**

```json
{
  "hits": [
    {
      "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
      "title": "My Awesome Game",
      "slug": "my-awesome-game",
      "synopsis": "An epic adventure awaits...",
      "engine_name": "Ren'Py",
      "like_count": 142,
      "average_score": 8.5
    }
  ],
  "total": 47,
  "page": 1,
  "per_page": 10,
  "processing_time_ms": 12
}
```

---

## Admin

All admin endpoints require the `admin` role.

| Method | Path | Description | Auth | Response |
|---|---|---|---|---|
| `GET` | `/admin/dashboard` | Aggregate platform stats | Admin | `JSON` |
| `GET` | `/admin/users` | User list with filters | Admin | `Paginated` |
| `PUT` | `/admin/users/{id}/role` | Change user role | Admin | `JSON` |
| `POST` | `/admin/users/{id}/ban` | Ban user | Admin | `JSON` |
| `POST` | `/admin/users/{id}/unban` | Unban user | Admin | `JSON` |
| `GET` | `/admin/games/pending` | Games awaiting approval | Admin | `Paginated` |
| `POST` | `/admin/games/{id}/approve` | Approve game | Admin | `JSON` |
| `POST` | `/admin/games/{id}/reject` | Reject game | Admin | `JSON` |
| `GET` | `/admin/reports` | Moderation queue | Admin/Mod | `Paginated` |
| `PUT` | `/admin/reports/{id}` | Resolve/dismiss report | Admin/Mod | `JSON` |
| `GET` | `/admin/audit-log` | System audit trail | Admin | `Paginated` |
| `DELETE` | `/admin/reviews/{id}` | Force-delete review | Admin/Mod | `204` |
| `DELETE` | `/admin/posts/{id}` | Force-delete post | Admin/Mod | `204` |
| `POST` | `/admin/tags` | Create tag | Admin | `JSON` |
| `PUT` | `/admin/tags/{id}` | Edit tag | Admin | `JSON` |
| `POST` | `/admin/engines` | Create engine | Admin | `JSON` |
| `PUT` | `/admin/engines/{id}` | Edit engine | Admin | `JSON` |
| `POST` | `/admin/forum-categories` | Create forum category | Admin | `JSON` |
| `PUT` | `/admin/forum-categories/{id}` | Edit forum category | Admin | `JSON` |

---

## Uploads

| Method | Path | Description | Auth | Response |
|---|---|---|---|---|
| `POST` | `/uploads/presign` | Generate presigned upload URL | Yes | `JSON` |

Uploads use a presigned URL flow — files go directly to MinIO, not through the API server.

**Upload limits:**

| Purpose | Max Size | Allowed Types |
|---|---|---|
| Avatar | 5 MB | JPEG, PNG, WebP, GIF |
| Screenshot | 10 MB | JPEG, PNG, WebP, GIF |
| Game file | 2 GB | ZIP, RAR, 7Z, EXE, APK |
| Forum attachment | 20 MB | JPEG, PNG, WebP, GIF, PDF, ZIP |

### Example: Presign Upload

**Request:**

```http
POST /api/v1/uploads/presign
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
Content-Type: application/json

{
  "filename": "screenshot-01.png",
  "content_type": "image/png",
  "purpose": "screenshot"
}
```

**Response `200`:**

```json
{
  "upload_url": "http://minio:9000/screenshots/abc123/screenshot-01.png?X-Amz-...",
  "object_key": "abc123/screenshot-01.png",
  "expires_in": 3600
}
```

After uploading to the presigned URL, confirm the upload via the relevant endpoint (e.g., `POST /games/{slug}/screenshots` with the `object_key`).

---

## Rate Limits

| Endpoint group | Limit |
|---|---|
| Auth (login, register) | 5 requests / 15 min per IP |
| General API | 100 requests / min per user |
| Search | 30 requests / min per IP |
| Chat messages | 1 message / 2 sec per user |
| File uploads | 10 presign requests / min per user |

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1627846260
```
