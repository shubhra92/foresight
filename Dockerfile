# ─────────────────────────────────────────────────────────────────────────────
# Project FORESIGHT — single image, three run modes (api / dashboard / refresh)
# Behavior is selected by the command (see docker-compose.yml).
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORESIGHT_DATA_DIR=/app/data/processed

# Non-root runtime user
RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

# 1) Dependencies first (cache layer — only rebuilt when requirements.txt changes)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 2) Application code
COPY --chown=app:app .streamlit ./.streamlit
COPY --chown=app:app src ./src
COPY --chown=app:app service ./service
COPY --chown=app:app app ./app

# 3) Data + artefacts. Named volumes in compose are primed from this image
#    content on first boot; run.sh rebuilds the pipeline if they go missing.
COPY --chown=app:app data ./data
COPY --chown=app:app models ./models
COPY --chown=app:app reports ./reports

COPY --chown=app:app run.sh ./
RUN chmod +x run.sh

USER app

EXPOSE 8000 8501

CMD ["./run.sh", "api"]