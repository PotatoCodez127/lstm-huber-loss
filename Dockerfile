# ==========================================
# Stage 1: Build & Dependency Consolidation
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies required for C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging configuration files
COPY pyproject.toml .

# Force wheel creation and download dependencies into a isolated directory
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir=/build/wheels .

# ==========================================
# Stage 2: Minimal Production Runtime
# ==========================================
FROM python:3.10-slim AS runtime

# Enforce system determinism: UTC timezone and unbuffered python logs
ENV TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Create a secure non-root operating user
RUN groupadd -g 10001 quant && \
    useradd -u 10001 -g quant -s /bin/bash -m quant

# Retrieve pre-compiled wheels from builder stage
COPY --from=builder /build/wheels /wheels

# Install finalized application dependencies explicitly
RUN pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy operational application files into the runtime frame
COPY data_engine_lstm.py model_lstm.py train_lstm.py test_lstm.py README.MD ./

# Enforce secure ownership permissions
RUN chown -R quant:quant /app
USER quant

# Expose verification capability via default execution target
CMD ["python", "train_lstm.py"]