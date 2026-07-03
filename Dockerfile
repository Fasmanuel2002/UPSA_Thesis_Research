FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        cmake \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency resolution from pyproject.toml + uv.lock
RUN pip install --no-cache-dir uv

# Copy dependency metadata first for better layer caching
COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev

# Copy application source
COPY . .

# Keep container ready for interactive use or manual script execution
CMD ["bash"]
