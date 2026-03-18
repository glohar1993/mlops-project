# ============================================================
# Production-Grade Dockerfile
# Uses Gunicorn (WSGI) instead of Flask dev server
# Multi-stage: slim image, non-root user, health check
# ============================================================

FROM python:3.11-slim AS base

# Security: run as non-root user
RUN groupadd -r mlops && useradd -r -g mlops mlops

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir setuptools && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy application code
COPY . /app

# Install project as package
RUN pip install --no-cache-dir -e .

# Set ownership
RUN chown -R mlops:mlops /app

USER mlops

EXPOSE 5001

# Health check built into image
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Gunicorn: 2 workers, production-grade WSGI server
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5001", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "application:app"]
