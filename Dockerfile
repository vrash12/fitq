# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# --- System deps (build + runtime) ---
# build-essential/g++ = compile dlib
# cmake + ninja-build = build system
# libgl1/libglib2.0-0 = common runtime deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    gcc \
    g++ \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# --- Hard check: fail early if cmake is missing ---
RUN which cmake && cmake --version

# --- Python tooling ---
# Install pip/setuptools/wheel first (dlib needs wheel builds)
# ALSO install pip's cmake/ninja to guarantee cmake exists in /usr/local/bin
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir cmake ninja \
    && which cmake && cmake --version
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- App code ---
COPY . /app

EXPOSE 8080

CMD exec gunicorn --bind :$PORT --workers 2 --threads 8 --timeout 0 wsgi:app
