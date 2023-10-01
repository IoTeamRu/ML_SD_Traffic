FROM python:3.10-slim as builder

RUN apt update && apt install -y make 
RUN pip install poetry
RUN mkdir -p /app
COPY . /app

WORKDIR /app
RUN poetry config virtualenvs.in-project true --local
RUN poetry install --no-interaction --no-root --only main

FROM python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install software-properties-common python3-launchpadlib -y
RUN add-apt-repository ppa:sumo/stable
RUN apt install sumo sumo-tools -y
COPY --from=builder /app /app

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
