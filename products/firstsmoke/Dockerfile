# Firstsmoke, for App Runner.
#
# Two stages. The builder renders the bundled incidents, which takes about a
# hundred seconds of OpenCV work and must not happen while a judge is waiting
# for the page; the runtime carries the rendered zips and none of the tooling
# that made them.
#
# OpenCV is pinned to 5.0.0.93 in three places that all have to agree: this
# file's constraint, the product's pyproject, and the assertion inside
# `visioncore` that raises on a 4.x wheel. 4.14.0 was released *after* 5.0.0, so
# an unpinned `pip install opencv-python` resolves to 4.x and the entry fails its
# single hard requirement while appearing to work.

# ---------------------------------------------------------------- builder ----
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY packages/visioncore /build/packages/visioncore
COPY packages/servicekit /build/packages/servicekit
COPY products/firstsmoke/pyproject.toml /build/products/firstsmoke/pyproject.toml
COPY products/firstsmoke/src /build/products/firstsmoke/src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir "opencv-python-headless==5.0.0.93" \
    && /opt/venv/bin/pip install --no-cache-dir \
        /build/packages/visioncore /build/packages/servicekit \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps /build/products/firstsmoke \
    && /opt/venv/bin/pip install --no-cache-dir \
        "fastapi==0.141.1" "uvicorn[standard]==0.53.0" "python-multipart==0.0.32" \
        "pydantic==2.13.5" "numpy==2.5.3" \
    && /opt/venv/bin/python -c "import cv2, sys; \
        assert cv2.__version__.startswith('5.'), f'OpenCV {cv2.__version__} is not 5.x'; \
        print('OpenCV', cv2.__version__)"

# Render the four bundled incidents into the image.
COPY products/firstsmoke/data /build/products/firstsmoke/data
RUN /opt/venv/bin/python -c "\
from pathlib import Path; \
from firstsmoke.scenarios import build_all; \
print([p.name for p in build_all(Path('/build/scenarios'))])"

# ---------------------------------------------------------------- runtime ----
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FIRSTSMOKE_SCENARIOS=/app/scenarios \
    OPENCV26_UPLOAD_DIR=/tmp/firstsmoke-uploads \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 watch

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/scenarios /app/scenarios
COPY products/firstsmoke/static /app/static
COPY products/firstsmoke/models /app/models
COPY products/firstsmoke/data/network /app/data/network

# The service resolves its static directory and its network metadata relative to
# the installed package, so point both at the copies above.
ENV FIRSTSMOKE_STATIC=/app/static \
    FIRSTSMOKE_MODELS=/app/models \
    FIRSTSMOKE_NETWORK=/app/data/network/hpwren_sites.js

WORKDIR /app
USER watch

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=4s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" || exit 1

# --log-level warning rather than --log-config /dev/null: uvicorn calls
# logging.config.fileConfig on whatever you hand it, and an empty file raises.
# servicekit configures the structured logger itself, so uvicorn only needs to
# stop duplicating every request line.
CMD ["sh", "-c", "exec uvicorn firstsmoke.service:app --host 0.0.0.0 --port ${PORT} --workers 1 --log-level warning --no-access-log"]
