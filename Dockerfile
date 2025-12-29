FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

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
