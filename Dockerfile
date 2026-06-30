# FraudShield AI — container image serving the REST API.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so they cache across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source.
COPY src ./src

# Train a model at build time so the image is ready to score out of the box.
RUN python -m src.train --n 20000

EXPOSE 8000

# Serve the FastAPI app.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
