# syntax=docker/dockerfile:1.7
#
# microholdem as a Chipzen bot. Build from the repository root, for the platform's linux/amd64:
#
#   docker build --platform=linux/amd64 -f integrations/chipzen/Dockerfile -t microholdem-chipzen:1 .
#   docker save microholdem-chipzen:1 | gzip > integrations/chipzen/microholdem-chipzen.tar.gz
#
# Stage 1 compiles microholdem's C++ extension into a wheel; stage 2 installs that wheel and the
# Chipzen SDK and runs bot.py as a non-root user. The platform adds a read-only root filesystem,
# seccomp, 0.5 vCPU and 256 MB on top; microholdem writes nothing at runtime.

# The SDK's Python starter base at python-v0.7.0. Tag: python:3.12-slim
ARG PYTHON_IMAGE=python:3.12-slim@sha256:dd29372629eeba2dd003fd9e9d35a5b8236c44727875a0364254b5127af88e65

# -----------------------------------------------------------------------------
# Stage 1: build the microholdem wheel (pybind11, C++17, -O3, no CPU-specific flags).
# -----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml setup.py README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

# -----------------------------------------------------------------------------
# Stage 2: runtime.
# -----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /bot

COPY integrations/chipzen/requirements.txt .
COPY --from=builder /wheels /tmp/wheels
RUN pip install -r requirements.txt /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels \
    && find /usr/local/lib/python3.12 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12 -type d -name tests -exec rm -rf {} + 2>/dev/null || true

COPY integrations/chipzen/bot.py /bot/bot.py

RUN groupadd --system --gid 10001 bot \
    && useradd --system --uid 10001 --gid bot --home-dir /bot --shell /usr/sbin/nologin bot \
    && chown -R bot:bot /bot
USER 10001

ENTRYPOINT ["python", "-c", "from bot import main; main()"]
