# backend/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Minimal runtime deps (keep only what you still need)
# - libglib2.0-0 + libgl1 are commonly needed by opencv-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app

EXPOSE 8080

# Run Cloud Run (make sure backend/wsgi.py exists and exposes `app`)
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
