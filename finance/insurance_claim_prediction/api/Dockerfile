# ── Build stage ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="realworld-ds-ml"
LABEL description="Insurance claim cost prediction API"
LABEL version="1.0.0"

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy API source
COPY main.py schema.py model.py features.py ./
COPY tests/ ./tests/

# Copy model artifacts
# The processed/ directory contains models.pkl, feature_spec.json etc.
# Mount this as a volume in production; copy it for a self-contained build.
COPY data/processed/ ./data/processed/

# Non-root user for security
RUN useradd -m apiuser && chown -R apiuser /app
USER apiuser

# Uvicorn listens on 8000
EXPOSE 8000

# Health-check: Docker polls /health every 30 seconds
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c \
  "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]