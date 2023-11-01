FROM python:3.10-slim

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y curl && \
    curl -L https://ollama.ai/download/ollama-linux-amd64 -o /usr/bin/ollama && \
    chmod +x /usr/bin/ollama

# Install python packages
COPY ./review/* /app
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "/app/entrypoint.py"]