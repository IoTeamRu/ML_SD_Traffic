# 
FROM python:3.10-slim as builder

RUN pip install poetry
RUN poetry config virtualenvs.in-project true --local
RUN mkdir -p /app
COPY . /app

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-root --only main

FROM python:3.10-slim as base

COPY --from=builder /app /app

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "app.main"]
