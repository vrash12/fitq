#backend/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# System deps for building dlib + runtime deps for opencv/mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    gcc \
    g++ \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade Python build tooling (do NOT install pip's cmake)
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# If a pip-shim cmake exists for any reason, remove it so /usr/bin/cmake is used
RUN rm -f /usr/local/bin/cmake /usr/local/bin/ninja || true

# Force PATH to prefer system binaries (cmake from apt)
ENV PATH="/usr/bin:${PATH}"
ENV CMAKE_EXECUTABLE="/usr/bin/cmake"

# Verify we are using the correct cmake
RUN which cmake && cmake --version

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8080
CMD exec gunicorn \
  --chdir backend \
  --bind 0.0.0.0:${PORT} \
  --workers 2 \
  --threads 8 \
  --timeout 0 \
  --access-logfile - \
  --error-logfile - \
  --log-level info \
  wsgi:app
