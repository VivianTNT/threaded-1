#!/usr/bin/env sh
set -eu

ART_DIR="${ART_DIR:-/app/recsys/artifacts}"
PORT="${PORT:-8000}"

required_files="
faiss_items_hm.joblib
user_vectors_hm.joblib
content_two_tower_hm.pt
"

for file in $required_files; do
  if [ ! -f "$ART_DIR/$file" ]; then
    echo "[start-api] Missing required artifact: $ART_DIR/$file" >&2
    exit 1
  fi
done

exec uvicorn recsys.api.main:app --host 0.0.0.0 --port "$PORT"
