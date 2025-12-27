# Use a slim Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cloud Run sets PORT (usually 8080)
ENV PORT=8080

WORKDIR /app

# (Optional) If you use packages that need compilation (psycopg2, etc.)
# uncomment these:
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential gcc \
#     && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code (including app/, wsgi.py, config.py, etc.)
COPY . /app

# Cloud Run listens on 8080 by default
EXPOSE 8080

# Run with Gunicorn (production server)
# If you want more workers, you can tune (2-4) depending on CPU/memory
CMD exec gunicorn --bind :$PORT --workers 2 --threads 8 --timeout 0 wsgi:app
