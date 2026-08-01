.PHONY: up down logs migrate makemigrations seed superuser test-backend test-frontend shell reset-db

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

migrate:
	docker compose exec backend alembic upgrade head

makemigrations:
	docker compose exec backend alembic revision --autogenerate -m "$(msg)"

seed:
	docker compose exec backend python scripts/seed_db.py

superuser:
	docker compose exec backend python scripts/create_superuser.py

test-backend:
	docker compose exec backend pytest -v

test-frontend:
	docker compose exec frontend npm test

shell:
	docker compose exec backend python -i

reset-db:
	docker compose down -v && $(MAKE) up && sleep 5 && $(MAKE) migrate && $(MAKE) seed
