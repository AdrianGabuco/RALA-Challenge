FROM python:3.11-slim


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libeccodes-dev \
        libopenjp2-7-dev \
        libnetcdf-dev \
        build-essential gcc gfortran \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY . /app


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


EXPOSE 8000

CMD ["gunicorn", "backend.wsgi:application", "--bind", "0.0.0.0:8000"]