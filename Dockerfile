FROM python:3.12-slim AS base

WORKDIR /app

# System deps for sentence-transformers, ChromaDB
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY fluxrag/ fluxrag/

RUN pip install --no-cache-dir .

# Default config mount point
VOLUME ["/etc/fluxrag"]

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

ENTRYPOINT ["fluxrag", "serve"]
CMD ["--config", "/etc/fluxrag/domain.yaml"]
