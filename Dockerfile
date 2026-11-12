# Stage 1 — build the frontend
FROM node:22-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend .
RUN npm run build

# Stage 2 — application
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY corpus corpus
COPY --from=frontend /fe/dist backend/static

WORKDIR /app/backend
EXPOSE 8000
# The embedding model (~90 MB) downloads on first start and is cached in the
# HF_HOME volume; the index is rebuilt from the corpus at boot (~1 s).
CMD ["sh", "-c", "python -m grounded.index_cli ../corpus --out data/index && uvicorn grounded.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
