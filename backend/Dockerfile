# Base image
FROM python:3.11-slim

# Install ecCodes system library
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libeccodes-dev build-essential gcc gfortran \
        libjpeg-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project
COPY . /app

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run server
CMD ["gunicorn", "backend.wsgi:application", "--bind", "0.0.0.0:8000"]