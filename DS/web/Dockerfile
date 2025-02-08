FROM python:3.12-slim

WORKDIR /app
EXPOSE 5000

WORKDIR /src
COPY ["web/", "./"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
