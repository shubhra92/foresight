"""
risk.py
=======
Project FORESIGHT — Risk Scoring & Decisioning
Client: NorthBay Living

Responsibilities
----------------
1. Stockout risk   — compare forecast demand over lead-time vs available stock.
2. Overstock risk  — compare on-hand stock vs forecast demand over a forward window.
3. Decisioning     — classify each SKU into one of four quadrants:
                       Reorder Now | Markdown/Clear | Watch/Volatile | Healthy
4. Financial impact — rupee sales-at-risk (stockout) and locked capital (overstock).
5. Action table    — structured output the ops team can act on immediately.

Risk scoring is fully transparent / rule-based — no black box.

Public API
----------
    from src.risk import run_risk_scoring

    risk_df = run_risk_scoring()   # returns per-SKU risk DataFrame
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds  (tunable without changing business logic)
# ─────────────────────────────────────────────────────────────────────────────

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning("Invalid value for %s='%s' — using default %.2f", name, raw, default)
        return default
    return value


# Stockout: if projected available stock covers < STOCKOUT_COVER_THRESHOLD
# weeks of forecast demand → HIGH stockout risk.
STOCKOUT_COVER_THRESHOLD: float = _env_float("FORESIGHT_STOCKOUT_THRESHOLD_WEEKS", 2.0)

# Overstock: if on-hand covers > OVERSTOCK_COVER_THRESHOLD weeks of demand
# → HIGH overstock risk.
OVERSTOCK_COVER_THRESHOLD: float = _env_float("FORESIGHT_OVERSTOCK_THRESHOLD_WEEKS", 8.0)

# Continuous risk score is normalised to [0, 1] using a sigmoid-like clip.
MAX_COVER_WEEKS: float = 16.0           # cap for normalisation


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load inputs
# ─────────────────────────────────────────────────────────────────────────────

def load_inputs(processed_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the weekly feature frame and the forward forecast.

    Returns
    -------
    (weekly_df, forecast_df)
    """
    wf_path = processed_dir / "weekly_features.parquet"
    fc_path = processed_dir / "forecast.parquet"

    if wf_path.exists():
        weekly_df = pd.read_parquet(wf_path)
    else:
        weekly_df = pd.read_csv(
            processed_dir / "weekly_features.csv",
            parse_dates=["week_start", "launch_date"],
        )

    if fc_path.exists():
        forecast_df = pd.read_parquet(fc_path)
    else:
        forecast_df = pd.read_csv(
            processed_dir / "forecast.csv",
            parse_dates=["week_start"],
        )

    weekly_df["week_start"]  = pd.to_datetime(weekly_df["week_start"])
    forecast_df["week_start"] = pd.to_datetime(forecast_df["week_start"])

    log.info("Loaded weekly_features: %s rows", f"{len(weekly_df):,}")
    log.info("Loaded forecast:        %s rows", f"{len(forecast_df):,}")

    return weekly_df, forecast_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Derive latest inventory snapshot per SKU
# ─────────────────────────────────────────────────────────────────────────────

def get_latest_inventory(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the most recent inventory position per SKU from the feature frame.

    Returns
    -------
    pd.DataFrame — one row per SKU:
        sku_id, on_hand_units, on_order_units, lead_time_days, reorder_point,
        available_stock, unit_cost, list_price, category, subcategory
    """
    agg_cols = [
        "on_hand_units", "on_order_units", "lead_time_days",
        "reorder_point", "available_stock", "unit_cost", "list_price",
        "category", "subcategory",
    ]
    # Take the last known row per SKU (most recent week)
    latest = (
        weekly_df.sort_values("week_start")
                 .groupby("sku_id")[agg_cols]
                 .last()
                 .reset_index()
    )
    return latest


# ─────────────────────────────────────────────────────────────────────────────
# 3. Aggregate forecast demand per SKU over different windows
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_forecast_demand(
    forecast_df: pd.DataFrame,
    lead_time_map: dict[str, int],
    horizon_weeks: int = 8,
) -> pd.DataFrame:
    """
    For each SKU compute:
        demand_over_lead_time   — Σ forecast over SKU's lead-time window.
        demand_over_horizon     — Σ forecast over the full horizon.
        avg_weekly_demand       — mean forecast across horizon.

    Parameters
    ----------
    forecast_df    : Forward forecast (sku_id, week_start, forecast).
    lead_time_map  : {sku_id: lead_time_days} dict.
    horizon_weeks  : Full horizon window.
    """
    records = []
    for sku_id, grp in forecast_df.groupby("sku_id"):
        grp = grp.sort_values("week_start").reset_index(drop=True)
        lead_days = lead_time_map.get(sku_id, 14)
        lead_weeks = max(1, int(np.ceil(lead_days / 7)))

        demand_lead  = float(grp.head(lead_weeks)["forecast"].sum())
        demand_horiz = float(grp.head(horizon_weeks)["forecast"].sum())
        avg_weekly   = float(grp.head(horizon_weeks)["forecast"].mean())

        records.append({
            "sku_id":              sku_id,
            "demand_over_lead":    demand_lead,
            "demand_over_horizon": demand_horiz,
            "avg_weekly_demand":   avg_weekly,
            "lead_weeks":          lead_weeks,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Compute raw risk scores [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_cover(cover_weeks: float, direction: str) -> float:
    """
    Convert weeks-of-cover into a [0, 1] risk score.

    For stockout  : low cover  → high risk.
    For overstock : high cover → high risk.
    """
    cover_weeks = max(0.0, min(cover_weeks, MAX_COVER_WEEKS))
    normalised = cover_weeks / MAX_COVER_WEEKS   # 0 = no cover; 1 = full cover

    if direction == "stockout":
        return round(1.0 - normalised, 4)
    else:  # overstock
        return round(normalised, 4)


def compute_risk_scores(
    inventory: pd.DataFrame,
    demand_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach stockout_risk and overstock_risk scores to each SKU.

    Stockout risk logic
    -------------------
    projected_stock  = on_hand + on_order
    cover_weeks      = projected_stock / max(avg_weekly_demand, 0.01)
    stockout_risk    = 1 - clip(cover_weeks / MAX_COVER_WEEKS, 0, 1)

    Overstock risk logic
    --------------------
    cover_weeks      = on_hand / max(avg_weekly_demand, 0.01)
    overstock_risk   = clip(cover_weeks / MAX_COVER_WEEKS, 0, 1)

    A cover of 0 weeks → stockout_risk = 1.0  (worst).
    A cover of 16+ weeks → overstock_risk = 1.0 (worst).
    """
    df = inventory.merge(demand_agg, on="sku_id", how="left")

    df["avg_weekly_demand"] = df["avg_weekly_demand"].fillna(0.01).clip(lower=0.01)

    # Stockout
    df["projected_stock"] = df["on_hand_units"] + df["on_order_units"]
    df["stockout_cover_weeks"] = (
        df["projected_stock"] / df["avg_weekly_demand"]
    ).clip(0, MAX_COVER_WEEKS)

    df["stockout_risk"] = df["stockout_cover_weeks"].apply(
        lambda c: _normalise_cover(c, "stockout")
    )

    # Overstock
    df["overstock_cover_weeks"] = (
        df["on_hand_units"] / df["avg_weekly_demand"]
    ).clip(0, MAX_COVER_WEEKS)

    df["overstock_risk"] = df["overstock_cover_weeks"].apply(
        lambda c: _normalise_cover(c, "overstock")
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Decisioning quadrants
# ─────────────────────────────────────────────────────────────────────────────

# Risk threshold — above this = "high" on that axis
RISK_HIGH_THRESHOLD: float = _env_float("FORESIGHT_RISK_HIGH_THRESHOLD", 0.5)


def assign_quadrant(stockout_risk: float, overstock_risk: float) -> str:
    """
    Map (stockout_risk, overstock_risk) → one of four quadrant labels.

    Quadrant logic (per brief §8.2):
        High stockout, Low overstock  → REORDER NOW
        High overstock, Low stockout  → MARKDOWN / CLEAR
        High on both                  → WATCH / VOLATILE
        Low on both                   → HEALTHY
    """
    high_so = stockout_risk  >= RISK_HIGH_THRESHOLD
    high_ov = overstock_risk >= RISK_HIGH_THRESHOLD

    if high_so and not high_ov:
        return "REORDER NOW"
    elif high_ov and not high_so:
        return "MARKDOWN / CLEAR"
    elif high_so and high_ov:
        return "WATCH / VOLATILE"
    else:
        return "HEALTHY"


QUADRANT_ACTIONS: dict[str, str] = {
    "REORDER NOW":      "Raise a replenishment order before stock runs out.",
    "MARKDOWN / CLEAR": "Promote or discount to free up working capital.",
    "WATCH / VOLATILE": "Investigate — demand is erratic. Review manually.",
    "HEALTHY":          "No action needed. Leave as is.",
}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Financial impact (rupee values)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rupee_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach rupee financial impact to each SKU.

    Sales at risk (stockout)
    ─────────────────────────
    Units expected to sell during the lead-time window but cannot be fulfilled
    if stock runs out:
        shortfall_units  = max(demand_over_lead - projected_stock, 0)
        sales_at_risk    = shortfall_units × list_price

    Locked capital (overstock)
    ──────────────────────────
    Inventory sitting beyond what will sell over the full horizon:
        excess_units     = max(on_hand - demand_over_horizon, 0)
        locked_capital   = excess_units × unit_cost

    Both represent the best-case avoidable loss with timely action.
    """
    df = df.copy()

    # Shortfall units = demand during lead time that can't be met
    df["shortfall_units"] = (
        df["demand_over_lead"] - df["projected_stock"]
    ).clip(lower=0)

    # Sales at risk = shortfall × list price
    df["sales_at_risk_inr"] = (
        df["shortfall_units"] * df["list_price"]
    ).round(2)

    # Excess units beyond what will sell over the forecast horizon
    df["excess_units"] = (
        df["on_hand_units"] - df["demand_over_horizon"]
    ).clip(lower=0)

    # Locked capital = excess inventory valued at cost
    df["locked_capital_inr"] = (
        df["excess_units"] * df["unit_cost"]
    ).round(2)

    # Total rupee at stake (used for bubble size in decisioning grid)
    df["rupee_at_stake"] = df["sales_at_risk_inr"] + df["locked_capital_inr"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Suggested reorder quantity
# ─────────────────────────────────────────────────────────────────────────────

def compute_reorder_qty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Suggest a replenishment order quantity for SKUs in 'REORDER NOW'.

    Formula:
        reorder_qty = demand_over_horizon - projected_stock + safety_buffer
        safety_buffer = avg_weekly_demand × lead_weeks  (one lead-time of safety)
    """
    df = df.copy()
    df["safety_buffer"] = (df["avg_weekly_demand"] * df["lead_weeks"]).clip(lower=0)

    df["suggested_reorder_qty"] = np.where(
        df["quadrant"] == "REORDER NOW",
        (df["demand_over_horizon"] - df["projected_stock"] + df["safety_buffer"])
        .clip(lower=0)
        .round()
        .astype(int),
        0,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. Assemble final risk table
# ─────────────────────────────────────────────────────────────────────────────

def build_risk_table(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Produce the final clean risk output table ordered by rupee_at_stake desc.

    Columns returned:
        sku_id, category, subcategory,
        stockout_risk, overstock_risk, quadrant, recommended_action,
        on_hand_units, projected_stock, demand_over_lead, demand_over_horizon,
        avg_weekly_demand, shortfall_units, excess_units,
        sales_at_risk_inr, locked_capital_inr, rupee_at_stake,
        suggested_reorder_qty, list_price, unit_cost
    """
    scored = scored.copy()
    scored["quadrant"] = scored.apply(
        lambda r: assign_quadrant(r["stockout_risk"], r["overstock_risk"]), axis=1
    )
    scored["recommended_action"] = scored["quadrant"].map(QUADRANT_ACTIONS)
    scored = compute_rupee_impact(scored)
    scored = compute_reorder_qty(scored)

    output_cols = [
        "sku_id", "category", "subcategory",
        "stockout_risk", "overstock_risk", "quadrant", "recommended_action",
        "on_hand_units", "on_order_units", "projected_stock",
        "demand_over_lead", "demand_over_horizon", "avg_weekly_demand",
        "shortfall_units", "excess_units",
        "sales_at_risk_inr", "locked_capital_inr", "rupee_at_stake",
        "suggested_reorder_qty", "list_price", "unit_cost",
        "stockout_cover_weeks", "overstock_cover_weeks",
        "lead_time_days", "reorder_point",
    ]
    available_cols = [c for c in output_cols if c in scored.columns]
    risk_df = scored[available_cols].copy()

    risk_df = risk_df.sort_values("rupee_at_stake", ascending=False).reset_index(drop=True)

    return risk_df


# ─────────────────────────────────────────────────────────────────────────────
# 9. Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def summarise_risk(risk_df: pd.DataFrame) -> dict:
    """
    Generate portfolio-level KPIs for the executive readout and dashboard.

    Returns
    -------
    dict with keys:
        total_skus, reorder_count, markdown_count,
        watch_count, healthy_count,
        total_sales_at_risk_inr, total_locked_capital_inr,
        quadrant_breakdown
    """
    qb = risk_df["quadrant"].value_counts().to_dict()
    return {
        "total_skus":                int(len(risk_df)),
        "reorder_count":             int(qb.get("REORDER NOW", 0)),
        "markdown_count":            int(qb.get("MARKDOWN / CLEAR", 0)),
        "watch_count":               int(qb.get("WATCH / VOLATILE", 0)),
        "healthy_count":             int(qb.get("HEALTHY", 0)),
        "total_sales_at_risk_inr":   round(float(risk_df["sales_at_risk_inr"].sum()), 2),
        "total_locked_capital_inr":  round(float(risk_df["locked_capital_inr"].sum()), 2),
        "quadrant_breakdown":        qb,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_risk_scoring(
    weekly_df: Optional[pd.DataFrame] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Execute the full FORESIGHT risk scoring pipeline.

    Steps
    -----
    1. Load weekly feature frame + forward forecast (if not provided).
    2. Extract latest inventory position per SKU.
    3. Aggregate forecast demand over lead-time and full horizon windows.
    4. Compute continuous stockout / overstock risk scores [0, 1].
    5. Classify into decisioning quadrants.
    6. Compute rupee financial impact.
    7. Build and return the final risk action table.
    8. Persist to data/processed/risk_scores.parquet (if save=True).

    Parameters
    ----------
    weekly_df   : Weekly feature frame from pipeline.py (optional).
    forecast_df : Forward forecast from forecast.py (optional).
    save        : Write risk output to disk.

    Returns
    -------
    pd.DataFrame — per-SKU risk table (see build_risk_table).
    """
    log.info("=" * 60)
    log.info("PROJECT FORESIGHT — Risk scoring pipeline starting")
    log.info("=" * 60)

    # 1. Load
    if weekly_df is None or forecast_df is None:
        weekly_df, forecast_df = load_inputs()

    weekly_df["week_start"]   = pd.to_datetime(weekly_df["week_start"])
    forecast_df["week_start"] = pd.to_datetime(forecast_df["week_start"])

    # 2. Latest inventory
    inventory = get_latest_inventory(weekly_df)
    log.info("Inventory snapshot: %d SKUs", len(inventory))

    # 3. Aggregate forecast demand
    lead_time_map = inventory.set_index("sku_id")["lead_time_days"].to_dict()
    horizon_weeks = forecast_df.groupby("sku_id")["week_start"].count().max()
    horizon_weeks = int(horizon_weeks) if pd.notna(horizon_weeks) else 8

    demand_agg = aggregate_forecast_demand(
        forecast_df, lead_time_map, horizon_weeks=horizon_weeks
    )
    log.info("Demand aggregated over %d-week horizon", horizon_weeks)

    # 4. Risk scores
    scored = compute_risk_scores(inventory, demand_agg)
    log.info("Risk scores computed")

    # 5–7. Quadrants, rupee impact, reorder qty
    risk_df = build_risk_table(scored)

    # 8. Persist
    if save:
        out_path = PROCESSED_DIR / "risk_scores.parquet"
        risk_df.to_parquet(out_path, index=False, engine="pyarrow")
        risk_df.to_csv(PROCESSED_DIR / "risk_scores.csv", index=False)
        log.info("Risk scores saved → %s", out_path)

    # Summary log
    summary = summarise_risk(risk_df)
    log.info("=" * 60)
    log.info("Risk scoring complete")
    log.info("  SKUs scored    : %d", summary["total_skus"])
    log.info("  REORDER NOW    : %d", summary["reorder_count"])
    log.info("  MARKDOWN/CLEAR : %d", summary["markdown_count"])
    log.info("  WATCH/VOLATILE : %d", summary["watch_count"])
    log.info("  HEALTHY        : %d", summary["healthy_count"])
    log.info("  Sales at risk  : ₹%s", f"{summary['total_sales_at_risk_inr']:,.0f}")
    log.info("  Locked capital : ₹%s", f"{summary['total_locked_capital_inr']:,.0f}")
    log.info("=" * 60)

    return risk_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    risk_df = run_risk_scoring()
    print("\n── Risk Table (top 10 by rupee at stake) ───────────────")
    print(
        risk_df[[
            "sku_id", "category", "quadrant",
            "stockout_risk", "overstock_risk",
            "sales_at_risk_inr", "locked_capital_inr", "rupee_at_stake",
            "recommended_action",
        ]].head(10).to_string(index=False)
    )
    print("\n── Portfolio Summary ────────────────────────────────────")
    import json
    print(json.dumps(summarise_risk(risk_df), indent=2))
