#!/usr/bin/env bash
# Convenience entrypoint for local development.
set -euo pipefail

# Wait for PG, run migrations, then start the API.
python -m alembic upgrade head
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
