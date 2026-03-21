#!/usr/bin/env sh
set -eu

PORT="${PORT:-8000}"

exec uvicorn recsys.api.main:app --host 0.0.0.0 --port "$PORT"
