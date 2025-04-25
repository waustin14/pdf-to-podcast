FROM python:3.9-slim

USER root
ARG DEBIAN_FRONTEND=noninteractive

RUN set -x \
    && apt-get update \
    && apt-get -y install espeak-ng gcc ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

RUN pip install --upgrade pip setuptools wheel

RUN pip install fastapi uvicorn kokoro>=0.3.1 misaki[en] pydub pydantic \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-requests \
    opentelemetry-instrumentation-redis \
    opentelemetry-exporter-otlp-proto-grpc \
    opentelemetry-instrumentation-httpx \
    opentelemetry-instrumentation-urllib3 \
    ujson

# Copy shared package first
COPY shared /shared
RUN pip install /shared

# Copy service files
COPY services/TTSProviderService/main.py ./

EXPOSE 8888

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]