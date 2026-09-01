FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRAUDSHIELD_MODEL_BUNDLE=/app/artifacts/model-bundle

RUN adduser --disabled-password --gecos "" appuser
WORKDIR /app

COPY --chown=appuser:appuser pyproject.toml README.md LICENSE ./
COPY --chown=appuser:appuser src ./src

RUN pip install --no-cache-dir . \
    && mkdir -p /app/artifacts \
    && chown -R appuser:appuser /app

EXPOSE 8000
USER appuser

CMD ["uvicorn", "fraudshield.api:app", "--host", "0.0.0.0", "--port", "8000"]
