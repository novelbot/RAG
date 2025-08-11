# =============================================================================
# Multi-stage Dockerfile for RAG Server
# =============================================================================
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# =============================================================================
# Production stage
# =============================================================================
FROM python:3.11-slim as production

# Install runtime system dependencies including ca-certificates for HTTPS
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r raguser && useradd -r -g raguser -d /app -s /bin/bash raguser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Make sure to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY --chown=raguser:raguser . .

# Create required directories including certs
RUN mkdir -p data/uploads data/temp logs certs && \
    chown -R raguser:raguser data logs certs

# Switch to non-root user
USER raguser

# Expose API ports (HTTP and HTTPS)
EXPOSE 8000 8443

# Health check (supports both HTTP and HTTPS based on SSL_ENABLED env var)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD sh -c 'if [ "$SSL_ENABLED" = "true" ]; then curl -k -f https://localhost:8443/health; else curl -f http://localhost:8000/health; fi' || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Default command
CMD ["python", "main.py"]

# =============================================================================
# Development stage (optional)
# =============================================================================
FROM builder as development

# Install development dependencies
RUN uv sync --frozen

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p data/uploads data/temp logs

# Expose API port and debug port
EXPOSE 8000 5678

# Set environment variables for development
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV DEBUG=true

# Development command with auto-reload
CMD ["uvicorn", "src.core.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]