FROM python:3.10-slim

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN echo 'Installing dependencies: ' && \
    apt update && apt install -y curl && \
    curl -L https://ollama.ai/download/ollama-linux-amd64 -o /usr/bin/ollama && \
    chmod +x /usr/bin/ollama

# Install python packages
COPY ./review/* /app
RUN chmod +x /app/run_server.bash && pip install --no-cache-dir -r requirements.txt && ls -la /app

ENTRYPOINT ["python", "/app/entrypoint.py"]
# ENTRYPOINT ["/app/run_server.bash"]