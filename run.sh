#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Project FORESIGHT — container entrypoint.
#
# Modes (selected by the first argument):
#   api        → FastAPI scoring service   (port FORESIGHT_PORT_API or 8000)
#   dashboard  → Streamlit planning dashboard (port FORESIGHT_PORT_DASH or 8501)
#   refresh    → one-shot weekly refresh + accuracy monitoring (src/run_weekly.py)
#
# On boot: rebuilds the pipeline (pipeline → forecast → risk) automatically if
# the processed artefacts are missing, or when REBUILD=1 is set (e.g. after the
# client drops new real data into data/raw).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NEED_REBUILD=0
for f in \
    data/processed/weekly_features.parquet \
    data/processed/forecast.parquet \
    data/processed/risk_scores.parquet; do
    [ -f "$f" ] || { NEED_REBUILD=1; echo "[foresight] missing $f"; }
done

if [ "${REBUILD:-0}" = "1" ] || [ "$NEED_REBUILD" = "1" ]; then
    echo "[foresight] Rebuilding pipeline (REBUILD=${REBUILD:-0}, missing=$NEED_REBUILD)…"
    python src/pipeline.py \
        && python src/forecast.py \
        && python src/risk.py \
        || { echo "[foresight] pipeline rebuild FAILED"; exit 1; }
    echo "[foresight] pipeline rebuilt OK"
fi

MODE="${1:-api}"
case "$MODE" in
    api)
        exec python -m uvicorn service.main:app --host 0.0.0.0 --port "${FORESIGHT_PORT_API:-8000}"
        ;;
    dashboard)
        exec python -m streamlit run app/app.py \
            --server.headless true \
            --server.address 0.0.0.0 \
            --server.port "${FORESIGHT_PORT_DASH:-8501}"
        ;;
    refresh)
        exec python src/run_weekly.py
        ;;
    *)
        echo "[foresight] unknown mode: $MODE (expected api|dashboard|refresh)" >&2
        exit 2
        ;;
esac