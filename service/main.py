"""
service/main.py
===============
Project FORESIGHT — Scoring Service (FastAPI)
Client: NorthBay Living

Endpoints
---------
GET  /health          Health check — confirms service is alive and data is loaded.
POST /predict         Return forecast + risk for a single SKU or a batch.
GET  /skus            List all available SKU IDs.
GET  /summary         Portfolio-level risk summary (KPIs).

Usage
-----
    uvicorn service.main:app --host 0.0.0.0 --port 8000 --reload

Or from the repo root:
    python -m uvicorn service.main:app --port 8000
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.risk import summarise_risk, assign_quadrant, QUADRANT_ACTIONS

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED = Path(os.environ.get("FORESIGHT_DATA_DIR", ROOT / "data" / "processed"))

# ─────────────────────────────────────────────────────────────────────────────
# Security configuration (enable via env in production)
# ─────────────────────────────────────────────────────────────────────────────

# Comma-separated allowed origins; default is local dev only.
CORS_ORIGINS: list[str] = [
    o.strip()
    for o in os.environ.get(
        "FORESIGHT_CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if o.strip()
]

# When set, data endpoints require `Authorization: Bearer <token>`.
# Leave unset for local/dev (open access), like the brief's demo setup.
AUTH_TOKEN: Optional[str] = os.environ.get("FORESIGHT_AUTH_TOKEN", "") or None


def require_auth(authorization: Optional[str] = Header(None)) -> None:
    """FastAPI dependency — enforce bearer token when configured."""
    if not AUTH_TOKEN:
        return  # auth disabled → open access (dev)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="Missing bearer token.")
    supplied = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, AUTH_TOKEN):
        raise HTTPException(401, detail="Invalid bearer token.")

# ─────────────────────────────────────────────────────────────────────────────
# App initialisation
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Project FORESIGHT — Scoring Service",
    description=(
        "Demand forecast and inventory risk API for NorthBay Living. "
        "Returns weekly SKU-level forecasts and stockout/overstock risk classifications."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Data store (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────

class _DataStore:
    forecast_df: Optional[pd.DataFrame] = None
    risk_df: Optional[pd.DataFrame] = None
    weekly_df: Optional[pd.DataFrame] = None
    backtest_metrics: Optional[dict] = None
    accuracy_history: list[dict] = []
    last_refresh: Optional[dict] = None
    ready: bool = False
    error: Optional[str] = None


_store = _DataStore()


def _load_data() -> None:
    """Load processed artefacts into the in-memory store at startup."""
    try:
        fc_path  = PROCESSED / "forecast.parquet"
        rk_path  = PROCESSED / "risk_scores.parquet"
        wf_path  = PROCESSED / "weekly_features.parquet"

        fc_csv   = PROCESSED / "forecast.csv"
        rk_csv   = PROCESSED / "risk_scores.csv"
        wf_csv   = PROCESSED / "weekly_features.csv"

        if fc_path.exists():
            _store.forecast_df = pd.read_parquet(fc_path)
        elif fc_csv.exists():
            _store.forecast_df = pd.read_csv(fc_csv, parse_dates=["week_start"])
        else:
            raise FileNotFoundError("forecast data not found")

        if rk_path.exists():
            _store.risk_df = pd.read_parquet(rk_path)
        elif rk_csv.exists():
            _store.risk_df = pd.read_csv(rk_csv)
        else:
            raise FileNotFoundError("risk_scores data not found")

        if wf_path.exists():
            _store.weekly_df = pd.read_parquet(wf_path)
        elif wf_csv.exists():
            _store.weekly_df = pd.read_csv(wf_csv, parse_dates=["week_start"])
        else:
            raise FileNotFoundError("weekly_features data not found")

        _store.forecast_df["week_start"] = pd.to_datetime(_store.forecast_df["week_start"])

        # Backtest evidence (WAPE vs baseline) persisted by src/forecast.py
        metrics_json = ROOT / "reports" / "backtest_metrics.json"
        if metrics_json.exists():
            _store.backtest_metrics = json.load(open(metrics_json))

        # Accuracy-monitoring history (appended by src/run_weekly.py)
        tracking = ROOT / "reports" / "accuracy_tracking.parquet"
        if tracking.exists():
            hist = pd.read_parquet(tracking)
            for col in ("wape_lgbm", "wape_baseline", "improvement_pct"):
                if col in hist.columns:
                    hist[col] = pd.to_numeric(hist[col], errors="coerce")
            rows = []
            for _, r in hist.iterrows():
                rows.append({
                    "run_at": str(pd.to_datetime(r.get("run_at"))),
                    "model_type": str(r.get("model_type", "")),
                    "forecast_end": str(pd.to_datetime(r.get("forecast_end"))),
                    "data_end_date": str(pd.to_datetime(r.get("data_end_date"))),
                    "wape_lgbm": None if pd.isna(r.get("wape_lgbm")) else float(r.get("wape_lgbm")),
                    "wape_baseline": None if pd.isna(r.get("wape_baseline")) else float(r.get("wape_baseline")),
                    "improvement_pct": None if pd.isna(r.get("improvement_pct")) else float(r.get("improvement_pct")),
                    "n_skus": int(r.get("n_skus", 0)),
                })
            _store.accuracy_history = rows
            log.info("Accuracy history loaded: %d snapshots", len(rows))

        last_refresh_json = ROOT / "reports" / "last_refresh.json"
        if last_refresh_json.exists():
            _store.last_refresh = json.load(open(last_refresh_json))

        _store.ready = True
        log.info("Data loaded: %d forecast rows, %d risk rows",
                 len(_store.forecast_df), len(_store.risk_df))

    except Exception as exc:
        _store.error = str(exc)
        log.error("Failed to load data: %s", exc)


@app.on_event("startup")
async def startup_event() -> None:
    _load_data()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    sku_ids: list[str] = Field(
        ...,
        min_length=1,
        description="One or more SKU IDs to score.",
        examples=[["SKU0001", "SKU0042"]],
    )
    horizon_weeks: int = Field(
        default=8,
        ge=1,
        le=26,
        description="Number of forecast weeks to return (1–26).",
    )

    @field_validator("sku_ids")
    @classmethod
    def skus_not_empty(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip().upper() for s in v if s.strip()]
        if not cleaned:
            raise ValueError("sku_ids must contain at least one non-empty value.")
        return cleaned


class WeeklyForecast(BaseModel):
    week_start: str
    forecast: float
    forecast_lo: float
    forecast_hi: float


class SKURiskResult(BaseModel):
    sku_id: str
    category: Optional[str]
    stockout_risk: float
    overstock_risk: float
    quadrant: str
    recommended_action: str
    sales_at_risk_inr: float
    locked_capital_inr: float
    rupee_at_stake: float
    suggested_reorder_qty: int
    on_hand_units: int
    avg_weekly_demand: float
    forecast: list[WeeklyForecast]
    model_type: Optional[str]


class PredictResponse(BaseModel):
    results: list[SKURiskResult]
    not_found: list[str]


class HealthResponse(BaseModel):
    status: str
    data_loaded: bool
    forecast_rows: Optional[int]
    risk_rows: Optional[int]
    error: Optional[str]
    last_refresh: Optional[dict] = None
    accuracy_tracking_rows: int = 0


class SummaryResponse(BaseModel):
    total_skus: int
    reorder_count: int
    markdown_count: int
    watch_count: int
    healthy_count: int
    total_sales_at_risk_inr: float
    total_locked_capital_inr: float
    quadrant_breakdown: dict


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build per-SKU result
# ─────────────────────────────────────────────────────────────────────────────

def _build_sku_result(
    sku_id: str,
    horizon_weeks: int,
) -> Optional[SKURiskResult]:
    """
    Assemble a SKURiskResult for a single SKU_ID.
    Returns None if the SKU is not found in either dataset.
    """
    fc_rows = _store.forecast_df[_store.forecast_df["sku_id"] == sku_id]
    rk_rows = _store.risk_df[_store.risk_df["sku_id"] == sku_id]

    if fc_rows.empty and rk_rows.empty:
        return None

    # ── Forecast slice ────────────────────────────────────────────────────────
    fc_slice = (
        fc_rows.sort_values("week_start")
               .head(horizon_weeks)
               .reset_index(drop=True)
    )

    weekly_forecast = [
        WeeklyForecast(
            week_start=str(row["week_start"].date()),
            forecast=round(float(row.get("forecast", 0)), 2),
            forecast_lo=round(float(row.get("forecast_lo", row.get("forecast", 0) * 0.75)), 2),
            forecast_hi=round(float(row.get("forecast_hi", row.get("forecast", 0) * 1.25)), 2),
        )
        for _, row in fc_slice.iterrows()
    ]

    model_type = str(fc_slice["model_type"].iloc[0]) if (
        not fc_slice.empty and "model_type" in fc_slice.columns
    ) else "unknown"

    # ── Risk data ─────────────────────────────────────────────────────────────
    def _safe(col: str, default=0.0):
        if rk_rows.empty or col not in rk_rows.columns:
            return default
        val = rk_rows.iloc[0][col]
        return default if pd.isna(val) else val

    stockout_risk  = float(_safe("stockout_risk",  0.0))
    overstock_risk = float(_safe("overstock_risk", 0.0))
    quadrant       = str(_safe("quadrant",         "HEALTHY"))
    recommended    = QUADRANT_ACTIONS.get(quadrant, "No action needed.")

    return SKURiskResult(
        sku_id=sku_id,
        category=str(_safe("category", "Unknown")),
        stockout_risk=round(stockout_risk, 4),
        overstock_risk=round(overstock_risk, 4),
        quadrant=quadrant,
        recommended_action=recommended,
        sales_at_risk_inr=round(float(_safe("sales_at_risk_inr", 0.0)), 2),
        locked_capital_inr=round(float(_safe("locked_capital_inr", 0.0)), 2),
        rupee_at_stake=round(float(_safe("rupee_at_stake", 0.0)), 2),
        suggested_reorder_qty=int(_safe("suggested_reorder_qty", 0)),
        on_hand_units=int(_safe("on_hand_units", 0)),
        avg_weekly_demand=round(float(_safe("avg_weekly_demand", 0.0)), 2),
        forecast=weekly_forecast,
        model_type=model_type,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Infrastructure"])
def health_check() -> HealthResponse:
    """
    Confirm the service is alive and processed data is loaded.
    Returns 200 when ready, 503 when data failed to load.
    """
    if not _store.ready:
        return HealthResponse(
            status="degraded",
            data_loaded=False,
            forecast_rows=None,
            risk_rows=None,
            error=_store.error,
        )
    return HealthResponse(
        status="ok",
        data_loaded=True,
        forecast_rows=len(_store.forecast_df),
        risk_rows=len(_store.risk_df),
        error=None,
        last_refresh=_store.last_refresh,
        accuracy_tracking_rows=len(_store.accuracy_history),
    )


@app.get("/skus", tags=["Data"], dependencies=[Depends(require_auth)])
def list_skus() -> dict:
    """Return all available SKU IDs and their categories."""
    if not _store.ready:
        raise HTTPException(503, detail="Service not ready — run pipeline first.")
    skus = (
        _store.risk_df[["sku_id", "category", "subcategory", "quadrant"]]
        .drop_duplicates("sku_id")
        .sort_values("sku_id")
        .to_dict(orient="records")
    )
    return {"total": len(skus), "skus": skus}


@app.get("/summary", response_model=SummaryResponse, tags=["Analytics"], dependencies=[Depends(require_auth)])
def portfolio_summary() -> SummaryResponse:
    """
    Portfolio-level KPIs: quadrant counts, total sales at risk,
    and total locked capital.
    """
    if not _store.ready:
        raise HTTPException(503, detail="Service not ready — run pipeline first.")
    s = summarise_risk(_store.risk_df)
    return SummaryResponse(**s)


@app.get("/evaluation", tags=["Analytics"], dependencies=[Depends(require_auth)])
def backtest_evaluation() -> dict:
    """
    Rolling-origin backtest evidence: WAPE vs seasonal-naive baseline,
    bias, MAE, and per-fold details — persisted by src/forecast.py.

    This is the honest head-to-head the planning dashboard surfaces:
    LightGBM is only shipped if it beats the baseline on backtest WAPE.
    """
    if not _store.ready:
        raise HTTPException(503, detail="Service not ready — run pipeline first.")
    if not _store.backtest_metrics:
        raise HTTPException(404, detail="No backtest metrics found. Run `python src/forecast.py` first.")
    payload = dict(_store.backtest_metrics)
    payload["history"] = _store.accuracy_history        # accuracy_tracking.parquet
    payload["last_refresh"] = _store.last_refresh       # last_refresh.json
    return payload


@app.post("/predict", response_model=PredictResponse, tags=["Scoring"], dependencies=[Depends(require_auth)])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Return demand forecast + inventory risk classification for one or more SKUs.

    Request body
    ------------
    ```json
    {
      "sku_ids": ["SKU0001", "SKU0042"],
      "horizon_weeks": 8
    }
    ```

    Response
    --------
    Per-SKU: weekly forecast values with 80% interval, risk scores,
    quadrant classification, recommended action, and rupee impact.

    Bad inputs (unknown SKU IDs, out-of-range horizon) are handled
    gracefully — unknown SKUs are listed in `not_found` rather than
    causing a 500 error.
    """
    if not _store.ready:
        raise HTTPException(503, detail="Service not ready — run pipeline first.")

    results: list[SKURiskResult] = []
    not_found: list[str] = []

    for sku_id in request.sku_ids:
        result = _build_sku_result(sku_id, request.horizon_weeks)
        if result is None:
            not_found.append(sku_id)
        else:
            results.append(result)

    return PredictResponse(results=results, not_found=not_found)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "service.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
