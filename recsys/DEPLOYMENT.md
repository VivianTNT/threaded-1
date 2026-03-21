# Recommender Deployment

This service is deployed separately from the Vercel web app.

## What gets deployed

- `recsys.api.main` as a standalone Python API
- Vercel keeps serving the Next.js app

## Why it must be separate

- Vercel is only deploying the Next app
- the recommender depends on large Python artifacts under `recsys/artifacts/`
- those artifacts are intentionally ignored by git and are not part of the Vercel deployment

## Required artifacts

At minimum, the service needs these files mounted at `recsys/artifacts/`:

- `faiss_items_hm.joblib`
- `content_two_tower_hm.pt`

This Render deployment disables the legacy `/recommend/user/{user_id}` path to keep memory use low.
If you later need that path, add `user_vectors_hm.joblib` and re-enable the endpoint.

## Docker deploy

Build from the repo root:

```bash
docker build -f recsys/Dockerfile -t threaded-recsys .
```

Run locally with the artifacts mounted:

```bash
docker run \
  -p 8000:8000 \
  -e PORT=8000 \
  -v "$PWD/recsys/artifacts:/app/recsys/artifacts" \
  threaded-recsys
```

## Render / Railway / Fly

Deploy the container from `recsys/Dockerfile`.

Mount a persistent disk or volume so the service has:

```text
/app/recsys/artifacts
```

Upload the recommender artifacts into that mounted directory before promoting traffic.

The container start command is already defined by `recsys/start-api.sh`.

## Vercel wiring

After the recommender is live, set this env var in the Vercel project:

```bash
RECSYS_API_URL=https://your-recommender-service.example.com
```

Redeploy the Vercel app after setting it.

## Verification

The recommender should expose:

```bash
GET /health
```

Example:

```bash
curl https://your-recommender-service.example.com/health
```

The web app homepage should then show the badge:

```text
FAISS + two-tower active
```
