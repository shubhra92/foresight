"""
Project FORESIGHT — weekly refresh + accuracy monitoring.

Runs the full stack (pipeline → forecast → risk), then appends the
model-vs-baseline accuracy snapshot to reports/accuracy_tracking.parquet
and writes reports/last_refresh.json so the dashboard and API can surface
"is the model degrading?" data rather than assuming it is.

Usage
-----
    python src/run_weekly.py            # full recompute + tracking append
    python src/run_weekly.py --dry-run  # recompute but do not persist tracking

Scheduling (example):
    cron.example ships a crontab entry, or, in Docker:
    docker compose run --rm refresh
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from pipeline import run_pipeline
from forecast import run_forecast
from risk import run_risk_scoring

log = logging.getLogger("foresight.weekly")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
TRACKING_PATH = REPORTS_DIR / "accuracy_tracking.parquet"
LAST_REFRESH_PATH = REPORTS_DIR / "last_refresh.json"


def load_accuracy_history() -> pd.DataFrame:
    """Return the persisted model-accuracy history (empty frame if none)."""
    if TRACKING_PATH.exists():
        return pd.read_parquet(TRACKING_PATH)
    return pd.DataFrame()


def append_accuracy_record(
    record: dict,
    save: bool = True,
) -> pd.DataFrame:
    """Append one WAPE/forecast snapshot to the tracking parquet."""
    hist = load_accuracy_history()
    row = pd.DataFrame([record])
    hist = pd.concat([hist, row], ignore_index=True)
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        hist.to_parquet(TRACKING_PATH, index=False, engine="pyarrow")
        log.info("Accuracy tracking appended → %s (%d rows)",
                 TRACKING_PATH, len(hist))
    return hist


def write_last_refresh(ok: bool, summary: dict) -> None:
    stamp = {
        "run_at": pd.Timestamp.now().isoformat(),
        "status": "ok" if ok else "failed",
        **summary,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LAST_REFRESH_PATH, "w") as fh:
        json.dump(stamp, fh, indent=2)


def run_weekly(save: bool = True) -> dict:
    log.info("=" * 60)
    log.info("PROJECT FORESIGHT — weekly refresh starting")
    log.info("=" * 60)

    weekly = run_pipeline(save=save)
    results = run_forecast(save=save)
    risk_df = run_risk_scoring(
        weekly_df=weekly,
        forecast_df=results["forecast_df"],
        save=save,
    )

    wc = results["wape_comparison"]
    forecast_dates = pd.to_datetime(results["forecast_df"]["week_start"])
    data_end = pd.to_datetime(weekly["week_start"]).max()

    record = {
        "run_at": pd.Timestamp.now(),
        "model_type": results["model_type"],
        "horizon_weeks": len(forecast_dates.unique()),
        "forecast_start": forecast_dates.min(),
        "forecast_end": forecast_dates.max(),
        "data_end_date": data_end,
        "wape_lgbm": wc["lgbm_wape"],
        "wape_baseline": wc["baseline_wape"],
        "improvement_pct": wc["improvement_pct"],
        "n_skus": int(len(risk_df)),
    }
    hist = append_accuracy_record(record, save=save)

    summary = {
        "model_type": record["model_type"],
        "wape_lgbm": record["wape_lgbm"],
        "wape_baseline": record["wape_baseline"],
        "improvement_pct": record["improvement_pct"],
        "data_end_date": record["data_end_date"].isoformat(),
        "forecast_end": record["forecast_end"].isoformat(),
        "n_skus": record["n_skus"],
        "tracking_rows": int(len(hist)),
    }
    write_last_refresh(ok=True, summary=summary)

    log.info("─" * 60)
    log.info("Weekly refresh complete | WAPE lgbm=%.4f baseline=%.4f | %d SKUs | tracking rows=%d",
             record["wape_lgbm"] or 0, record["wape_baseline"] or 0,
             record["n_skus"], len(hist))
    log.info("─" * 60)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FORESIGHT weekly refresh + accuracy tracking")
    parser.add_argument("--dry-run", action="store_true",
                        help="recompute everything but do not persist tracking")
    args = parser.parse_args()
    run_weekly(save=not args.dry_run)