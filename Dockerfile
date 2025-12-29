# backend/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Only keep runtime libs you still need (remove these too if you no longer use opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# This copies EVERYTHING inside backend/ into /app
COPY . /app

EXPOSE 8080

# IMPORTANT: no --chdir backend here
CMD exec gunicorn \
  --bind 0.0.0.0:${PORT} \
  --workers 2 \
  --threads 8 \
  --timeout 0 \
  --access-logfile - \
  --error-logfile - \
  --log-level info \
  wsgi:app
