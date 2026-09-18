"""
pipeline.py
===========
Project FORESIGHT — Data Pipeline
Client: NorthBay Living

Responsibilities
----------------
1. Ingest       — load the four raw CSV extracts.
2. Validate     — report data-quality issues (nulls, dupes, bad values).
3. Clean        — programmatic fixes: dedup, imputation, type coercion,
                  label normalisation.
4. Merge        — join fact + dimensions into one enriched daily frame.
5. Aggregate    — roll up from daily → weekly SKU-level demand.
6. Feature eng  — lags (1 / 2 / 4 wks), rolling stats, calendar &
                  promotion signals ready for the forecasting model.
7. Persist      — write the analysis-ready dataset to data/processed/.

Public API
----------
    from src.pipeline import run_pipeline

    weekly_df = run_pipeline()          # full pipeline, returns weekly frame
    weekly_df = run_pipeline(save=False) # skip writing to disk

Everything is reproducible end-to-end from raw CSVs with one call.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]   # foresight/
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ingestion
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_tables(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """
    Load the four raw CSV extracts from *raw_dir*.

    Returns
    -------
    dict with keys: 'sales', 'sku_master', 'calendar', 'inventory'
    """
    files = {
        "sales":     "sales_daily.csv",
        "sku_master": "sku_master.csv",
        "calendar":  "calendar.csv",
        "inventory": "inventory_snapshots.csv",
    }

    tables: dict[str, pd.DataFrame] = {}
    for key, fname in files.items():
        path = raw_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Raw file not found: {path}\n"
                "Run `python data/generate_data.py` first."
            )
        tables[key] = pd.read_csv(path, low_memory=False)
        log.info("Loaded %-20s — %s rows, %s cols",
                 fname, f"{len(tables[key]):,}", tables[key].shape[1])

    return tables


# ─────────────────────────────────────────────────────────────────────────────
# 2. Validation — report issues BEFORE cleaning
# ─────────────────────────────────────────────────────────────────────────────

def validate_tables(tables: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """
    Inspect each table for common data-quality problems and return a
    structured report (does not modify the data).

    Issues checked
    --------------
    - Missing value counts per column
    - Duplicate row counts
    - Negative numeric values (unit_price, units_sold, on_hand_units)
    - Referential integrity: all sku_ids in sales/inventory present in sku_master
    - Date range plausibility
    """
    report: dict[str, dict] = {}

    # ── sales_daily ──────────────────────────────────────────────────────────
    s = tables["sales"].copy()
    report["sales"] = {
        "rows": len(s),
        "nulls": s.isnull().sum().to_dict(),
        "duplicates": int(s.duplicated().sum()),
        "negative_unit_price": int((s["unit_price"] < 0).sum()),
        "negative_units_sold": int((s["units_sold"] < 0).sum()),
        "date_range": (str(s["date"].min()), str(s["date"].max())),
    }

    # ── sku_master ────────────────────────────────────────────────────────────
    m = tables["sku_master"].copy()
    report["sku_master"] = {
        "rows": len(m),
        "nulls": m.isnull().sum().to_dict(),
        "duplicates": int(m.duplicated().sum()),
        "unique_skus": int(m["sku_id"].nunique()),
        "category_labels": sorted(m["category"].unique().tolist()),
    }

    # ── calendar ─────────────────────────────────────────────────────────────
    c = tables["calendar"].copy()
    report["calendar"] = {
        "rows": len(c),
        "nulls": c.isnull().sum().to_dict(),
        "duplicates": int(c.duplicated().sum()),
        "date_range": (str(c["date"].min()), str(c["date"].max())),
    }

    # ── inventory_snapshots ──────────────────────────────────────────────────
    inv = tables["inventory"].copy()
    report["inventory"] = {
        "rows": len(inv),
        "nulls": inv.isnull().sum().to_dict(),
        "duplicates": int(inv.duplicated().sum()),
        "negative_on_hand": int((inv["on_hand_units"] < 0).sum()),
    }

    # ── Referential integrity ─────────────────────────────────────────────────
    master_skus = set(tables["sku_master"]["sku_id"])
    orphan_sales = set(tables["sales"]["sku_id"]) - master_skus
    orphan_inv = set(tables["inventory"]["sku_id"]) - master_skus
    report["referential_integrity"] = {
        "orphan_skus_in_sales": sorted(orphan_sales),
        "orphan_skus_in_inventory": sorted(orphan_inv),
    }

    # Log a summary
    for tbl, issues in report.items():
        if tbl == "referential_integrity":
            continue
        null_total = sum(v for v in issues.get("nulls", {}).values()
                         if isinstance(v, (int, float)))
        log.info(
            "Validate %-20s | rows=%s | nulls=%d | dupes=%d",
            tbl,
            f"{issues['rows']:,}",
            null_total,
            issues.get("duplicates", 0),
        )

    return report


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean sales_daily.

    Steps (each documented for audit trail):
    1. Parse date column to datetime.
    2. Remove exact duplicate rows (same date + sku_id + all values).
    3. Fix negative unit_price → absolute value (sign error, not a return).
    4. Impute missing units_sold with 0 (no sale record = no sale that day).
    5. Impute missing revenue = units_sold × unit_price where possible,
       else 0.
    6. Clip units_sold to non-negative (safety guard).
    7. Cast types.
    """
    df = df.copy()

    # 1. Date parsing
    df["date"] = pd.to_datetime(df["date"])

    # 2. Remove exact duplicates (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset=["date", "sku_id", "units_sold", "revenue",
                                     "unit_price", "promo_flag"])
    log.info("clean_sales  | dropped %d exact duplicates", before - len(df))

    # 3. Fix negative unit_price
    neg_mask = df["unit_price"] < 0
    df.loc[neg_mask, "unit_price"] = df.loc[neg_mask, "unit_price"].abs()
    log.info("clean_sales  | corrected %d negative unit_price rows", neg_mask.sum())

    # 4. Impute missing units_sold → 0
    df["units_sold"] = df["units_sold"].fillna(0)

    # 5. Impute missing revenue
    missing_rev = df["revenue"].isna()
    df.loc[missing_rev, "revenue"] = (
        df.loc[missing_rev, "units_sold"] * df.loc[missing_rev, "unit_price"]
    )
    df["revenue"] = df["revenue"].fillna(0)
    log.info("clean_sales  | imputed %d missing revenue values", missing_rev.sum())

    # 6. Clip negatives
    df["units_sold"] = df["units_sold"].clip(lower=0)
    df["revenue"] = df["revenue"].clip(lower=0)

    # 7. Types
    df["units_sold"] = df["units_sold"].astype(int)
    df["promo_flag"] = df["promo_flag"].fillna(0).astype(int)
    df["unit_price"] = df["unit_price"].astype(float)
    df["revenue"] = df["revenue"].astype(float)
    df["sku_id"] = df["sku_id"].astype(str).str.strip()

    return df.sort_values(["date", "sku_id"]).reset_index(drop=True)


def clean_sku_master(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean sku_master.

    Steps:
    1. Normalise category / subcategory labels → title-case, strip whitespace.
    2. Parse launch_date to datetime.
    3. Remove duplicate sku_ids (keep first).
    4. Cast unit_cost and list_price to float.
    """
    df = df.copy()

    # 1. Normalise text labels
    df["category"] = df["category"].str.strip().str.title()
    df["subcategory"] = df["subcategory"].str.strip().str.title()
    log.info("clean_sku_master | normalised category/subcategory labels")

    # 2. Date parsing
    df["launch_date"] = pd.to_datetime(df["launch_date"])

    # 3. Deduplicate on sku_id
    before = len(df)
    df = df.drop_duplicates(subset=["sku_id"], keep="first")
    log.info("clean_sku_master | dropped %d duplicate sku_ids", before - len(df))

    # 4. Cast
    df["unit_cost"] = df["unit_cost"].astype(float)
    df["list_price"] = df["list_price"].astype(float)
    df["sku_id"] = df["sku_id"].astype(str).str.strip()

    return df.reset_index(drop=True)


def clean_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean calendar.

    Steps:
    1. Parse date to datetime.
    2. Fill null promo_event with 'None'.
    3. Ensure is_holiday is integer.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["promo_event"] = df["promo_event"].fillna("None")
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    df["season"] = df["season"].str.strip().str.title()
    return df.reset_index(drop=True)


def clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean inventory_snapshots.

    Steps:
    1. Parse date to datetime.
    2. Clip on_hand_units / on_order_units to non-negative.
    3. Fill missing lead_time with median; reorder_point with median.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["on_hand_units"] = df["on_hand_units"].clip(lower=0).fillna(0).astype(int)
    df["on_order_units"] = df["on_order_units"].clip(lower=0).fillna(0).astype(int)

    median_lt = df["lead_time_days"].median()
    median_rp = df["reorder_point"].median()
    df["lead_time_days"] = df["lead_time_days"].fillna(median_lt).astype(int)
    df["reorder_point"] = df["reorder_point"].fillna(median_rp).astype(int)
    df["sku_id"] = df["sku_id"].astype(str).str.strip()

    return df.reset_index(drop=True)


def clean_all(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply all cleaning functions and return a dict of clean DataFrames."""
    return {
        "sales":      clean_sales(tables["sales"]),
        "sku_master": clean_sku_master(tables["sku_master"]),
        "calendar":   clean_calendar(tables["calendar"]),
        "inventory":  clean_inventory(tables["inventory"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Merge — enrich daily sales with dimension attributes
# ─────────────────────────────────────────────────────────────────────────────

def merge_tables(clean: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join fact + dimensions into one enriched daily frame.

    Join logic
    ----------
    sales_daily
        LEFT JOIN sku_master   ON sku_id
        LEFT JOIN calendar     ON date
        LEFT JOIN inventory_snapshots ON (sku_id, date nearest-weekly)

    The inventory table is weekly; we forward-fill the latest snapshot
    to every sales day within the same week.

    Returns
    -------
    pd.DataFrame  — one row per SKU per day, enriched.
    """
    sales = clean["sales"]
    sku_master = clean["sku_master"]
    calendar = clean["calendar"]
    inventory = clean["inventory"]

    # ── sales × sku_master ───────────────────────────────────────────────────
    df = sales.merge(
        sku_master[["sku_id", "category", "subcategory",
                    "launch_date", "unit_cost", "list_price"]],
        on="sku_id", how="left"
    )
    log.info("merge | after sku_master join: %s rows", f"{len(df):,}")

    # ── × calendar ───────────────────────────────────────────────────────────
    df = df.merge(
        calendar[["date", "week", "month", "year", "season",
                  "is_holiday", "promo_event", "day_of_week"]],
        on="date", how="left"
    )
    log.info("merge | after calendar join: %s rows", f"{len(df):,}")

    # ── × inventory (forward-fill weekly snapshot to daily) ──────────────────
    # Set the inventory snapshot date as the Monday of that ISO-week so
    # we can merge on (sku_id, iso_week_start).
    inv = inventory.copy()
    inv["week_start"] = inv["date"] - pd.to_timedelta(
        inv["date"].dt.dayofweek, unit="D"
    )

    # For each sales row, compute its week_start
    df["week_start"] = df["date"] - pd.to_timedelta(
        df["date"].dt.dayofweek, unit="D"
    )

    # Keep the LAST snapshot per (sku_id, week_start)
    inv_weekly = (
        inv.sort_values("date")
           .groupby(["sku_id", "week_start"], as_index=False)
           .last()[["sku_id", "week_start", "on_hand_units",
                    "on_order_units", "lead_time_days", "reorder_point"]]
    )

    df = df.merge(inv_weekly, on=["sku_id", "week_start"], how="left")

    # Forward-fill inventory fields for SKUs with missing snapshots
    inv_cols = ["on_hand_units", "on_order_units", "lead_time_days", "reorder_point"]
    df = df.sort_values(["sku_id", "date"])
    df[inv_cols] = (
        df.groupby("sku_id")[inv_cols]
          .transform(lambda s: s.ffill().bfill())
    )

    # Final fallback: fill any remaining nulls with safe defaults
    df["on_hand_units"] = df["on_hand_units"].fillna(0).astype(int)
    df["on_order_units"] = df["on_order_units"].fillna(0).astype(int)
    df["lead_time_days"] = df["lead_time_days"].fillna(14).astype(int)
    df["reorder_point"] = df["reorder_point"].fillna(30).astype(int)

    df = df.drop(columns=["week_start"]).reset_index(drop=True)
    log.info("merge | enriched daily frame: %s rows × %s cols",
             f"{len(df):,}", df.shape[1])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Aggregate daily → weekly SKU-level demand
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the enriched daily frame to weekly SKU-level demand.

    Weekly grain: ISO week starting Monday.
    Numeric columns are summed (units, revenue) or last-value taken
    (inventory snapshots, price).

    Returns
    -------
    pd.DataFrame  — columns:
        week_start, sku_id, category, subcategory, units_sold, revenue,
        avg_unit_price, promo_weeks (binary), is_holiday_week (binary),
        season, on_hand_units, on_order_units, lead_time_days, reorder_point,
        unit_cost, list_price, launch_date
    """
    df = daily.copy()
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")

    agg = df.groupby(["week_start", "sku_id"]).agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        avg_unit_price=("unit_price", "mean"),
        promo_weeks=("promo_flag", "max"),       # 1 if any promo day that week
        is_holiday_week=("is_holiday", "max"),   # 1 if any holiday that week
        season=("season", "last"),
        category=("category", "last"),
        subcategory=("subcategory", "last"),
        on_hand_units=("on_hand_units", "last"),
        on_order_units=("on_order_units", "last"),
        lead_time_days=("lead_time_days", "last"),
        reorder_point=("reorder_point", "last"),
        unit_cost=("unit_cost", "last"),
        list_price=("list_price", "last"),
        launch_date=("launch_date", "last"),
    ).reset_index()

    agg = agg.sort_values(["sku_id", "week_start"]).reset_index(drop=True)
    log.info("aggregate_weekly | weekly frame: %s rows × %s cols",
             f"{len(agg):,}", agg.shape[1])

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer model-ready features from the weekly SKU frame.

    Features created (all computed strictly from the PAST — no leakage)
    ───────────────────────────────────────────────────────────────────
    Lag features (per SKU, sorted by week):
        lag_1w      — units_sold 1 week ago
        lag_2w      — units_sold 2 weeks ago
        lag_4w      — units_sold 4 weeks ago  (same-week, prior month)
        lag_52w     — units_sold 52 weeks ago (same-week, prior year)

    Rolling statistics (computed on lag_1w onward, not current row):
        roll_mean_4w  — 4-week rolling mean of units_sold
        roll_std_4w   — 4-week rolling std  of units_sold
        roll_mean_8w  — 8-week rolling mean of units_sold

    Calendar / seasonality:
        week_of_year       — ISO week number (1–53)
        month              — month of year  (1–12)
        is_q4              — binary: Q4 (Oct–Dec) flag (peak season)
        season_encoded     — ordinal encoding of season string

    Price & promotion:
        price_ratio        — avg_unit_price / list_price  (discount depth)
        promo_weeks        — already present (binary)
        is_holiday_week    — already present (binary)

    Inventory position:
        stock_cover_weeks  — on_hand_units / max(roll_mean_4w, 1)
                             how many weeks of stock at current run-rate
        available_stock    — on_hand_units + on_order_units

    SKU age:
        sku_age_weeks      — weeks since launch_date

    Notes
    -----
    * All lag/rolling ops use groupby(sku_id) + shift(n) so week boundaries
      are respected per SKU.
    * Rows with insufficient history for lags will have NaN; the model
      handles this — we do NOT forward-fill lag features as that would
      constitute leakage.
    """
    df = weekly.copy().sort_values(["sku_id", "week_start"]).reset_index(drop=True)

    grp = df.groupby("sku_id")["units_sold"]

    # ── Lag features ─────────────────────────────────────────────────────────
    df["lag_1w"] = grp.shift(1)
    df["lag_2w"] = grp.shift(2)
    df["lag_4w"] = grp.shift(4)
    df["lag_52w"] = grp.shift(52)

    # ── Rolling statistics (shift first to avoid current-row leakage) ────────
    shifted = grp.shift(1)  # start roll from last week
    df["roll_mean_4w"] = (
        shifted.groupby(df["sku_id"])
               .transform(lambda s: s.rolling(4, min_periods=1).mean())
    )
    df["roll_std_4w"] = (
        shifted.groupby(df["sku_id"])
               .transform(lambda s: s.rolling(4, min_periods=2).std().fillna(0))
    )
    df["roll_mean_8w"] = (
        shifted.groupby(df["sku_id"])
               .transform(lambda s: s.rolling(8, min_periods=1).mean())
    )

    # ── Calendar features ─────────────────────────────────────────────────────
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    df["is_q4"] = df["month"].isin([10, 11, 12]).astype(int)

    season_order = {"Spring": 0, "Summer": 1, "Monsoon": 2, "Autumn": 3, "Winter": 4}
    df["season_encoded"] = df["season"].map(season_order).fillna(2).astype(int)

    # ── Price & promotion ─────────────────────────────────────────────────────
    df["price_ratio"] = (df["avg_unit_price"] / df["list_price"].replace(0, np.nan)
                         ).fillna(1.0).clip(0.5, 1.5)

    # ── Inventory features ────────────────────────────────────────────────────
    df["available_stock"] = df["on_hand_units"] + df["on_order_units"]
    df["stock_cover_weeks"] = (
        df["on_hand_units"] / df["roll_mean_4w"].replace(0, np.nan)
    ).fillna(0).clip(upper=52)

    # ── SKU age ───────────────────────────────────────────────────────────────
    df["sku_age_weeks"] = (
        (df["week_start"] - df["launch_date"]).dt.days / 7
    ).clip(lower=0).fillna(0).astype(int)

    # ── Category dummies ──────────────────────────────────────────────────────
    # Keep as a column for dashboard filtering; model uses encoded version
    cat_dummies = pd.get_dummies(df["category"], prefix="cat", drop_first=False)
    cat_dummies = cat_dummies.astype(int)
    df = pd.concat([df, cat_dummies], axis=1)

    log.info("engineer_features | feature frame: %s rows × %s cols",
             f"{len(df):,}", df.shape[1])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Persist
# ─────────────────────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, processed_dir: Path = PROCESSED_DIR) -> Path:
    """
    Persist the analysis-ready dataset as a Parquet file for fast I/O.
    Also writes a CSV copy for inspection.

    Returns the path to the Parquet file.
    """
    parquet_path = processed_dir / "weekly_features.parquet"
    csv_path = processed_dir / "weekly_features.csv"

    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    df.to_csv(csv_path, index=False)

    log.info("Saved processed data → %s  (%s rows)", parquet_path, f"{len(df):,}")
    return parquet_path


# ─────────────────────────────────────────────────────────────────────────────
# Data Quality Report — written to reports/
# ─────────────────────────────────────────────────────────────────────────────

def write_dq_report(
    validation_report: dict[str, dict],
    processed: pd.DataFrame,
    reports_dir: Optional[Path] = None,
) -> Path:
    """
    Write a plain-text data-quality report summarising validation findings
    and post-cleaning shape to reports/data_quality_report.txt.
    """
    if reports_dir is None:
        reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_path = reports_dir / "data_quality_report.txt"

    lines: list[str] = [
        "=" * 70,
        "PROJECT FORESIGHT — Data Quality Report",
        "Client: NorthBay Living",
        "=" * 70,
        "",
        "── RAW TABLE STATISTICS (before cleaning) ──────────────────────────",
        "",
    ]

    for tbl, info in validation_report.items():
        if tbl == "referential_integrity":
            continue
        lines.append(f"  Table: {tbl}")
        lines.append(f"    Rows       : {info.get('rows', 'N/A'):,}")
        lines.append(f"    Duplicates : {info.get('duplicates', 0)}")
        nulls = info.get("nulls", {})
        total_nulls = sum(v for v in nulls.values() if isinstance(v, (int, float)))
        lines.append(f"    Total nulls: {total_nulls}")
        null_detail = {k: v for k, v in nulls.items() if v > 0}
        if null_detail:
            lines.append(f"    Null cols  : {null_detail}")
        if "negative_unit_price" in info and info["negative_unit_price"] > 0:
            lines.append(f"    ⚠ Negative unit_price: {info['negative_unit_price']} rows")
        if "category_labels" in info:
            lines.append(f"    Category labels (raw): {info['category_labels']}")
        lines.append("")

    ri = validation_report.get("referential_integrity", {})
    lines += [
        "── REFERENTIAL INTEGRITY ───────────────────────────────────────────",
        "",
        f"  Orphan SKUs in sales (not in sku_master): "
        f"{len(ri.get('orphan_skus_in_sales', []))}",
        f"  Orphan SKUs in inventory (not in sku_master): "
        f"{len(ri.get('orphan_skus_in_inventory', []))}",
        "",
        "── CLEANING DECISIONS ──────────────────────────────────────────────",
        "",
        "  sales_daily:",
        "    • Exact duplicate rows removed (keep first).",
        "    • Negative unit_price corrected to absolute value (sign error).",
        "    • Missing units_sold imputed with 0 (no-sale days).",
        "    • Missing revenue imputed as units_sold × unit_price, else 0.",
        "    • units_sold and revenue clipped to ≥ 0.",
        "",
        "  sku_master:",
        "    • category and subcategory normalised to title-case, stripped.",
        "    • Duplicate sku_ids removed (keep first).",
        "",
        "  calendar:",
        "    • Null promo_event filled with 'None'.",
        "    • is_holiday cast to int.",
        "",
        "  inventory_snapshots:",
        "    • on_hand / on_order clipped to ≥ 0.",
        "    • Missing lead_time / reorder_point filled with column median.",
        "",
        "── POST-PIPELINE ANALYSIS-READY DATASET ────────────────────────────",
        "",
        f"  Shape      : {processed.shape[0]:,} rows × {processed.shape[1]} columns",
        f"  SKUs       : {processed['sku_id'].nunique():,}",
        f"  Date range : {processed['week_start'].min().date()} → "
        f"{processed['week_start'].max().date()}",
        f"  Null count : {processed.isnull().sum().sum()}",
        "",
        "  Feature columns:",
    ]
    for col in sorted(processed.columns):
        null_n = int(processed[col].isnull().sum())
        dtype = str(processed[col].dtype)
        lines.append(f"    {col:<30} {dtype:<12} nulls={null_n}")

    lines += ["", "=" * 70, "End of report.", "=" * 70]

    out_path.write_text("\n".join(lines))
    log.info("DQ report written → %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    raw_dir: Path = RAW_DIR,
    save: bool = True,
    write_report: bool = True,
) -> pd.DataFrame:
    """
    Execute the full FORESIGHT data pipeline end-to-end.

    Steps
    -----
    1. Load raw CSVs
    2. Validate (report issues)
    3. Clean all tables
    4. Merge into enriched daily frame
    5. Aggregate to weekly grain
    6. Engineer features
    7. Persist to data/processed/  (if save=True)
    8. Write DQ report             (if write_report=True)

    Parameters
    ----------
    raw_dir      : Path to the directory containing raw CSVs.
    save         : If True, write the processed dataset to disk.
    write_report : If True, write the data-quality report to reports/.

    Returns
    -------
    pd.DataFrame — weekly SKU-level feature frame, analysis-ready.
    """
    log.info("=" * 60)
    log.info("PROJECT FORESIGHT — Pipeline starting")
    log.info("=" * 60)

    # 1. Load
    tables = load_raw_tables(raw_dir)

    # 2. Validate
    dq_report = validate_tables(tables)

    # 3. Clean
    clean = clean_all(tables)

    # 4. Merge
    daily_enriched = merge_tables(clean)

    # 5. Aggregate
    weekly = aggregate_weekly(daily_enriched)

    # 6. Feature engineering
    features = engineer_features(weekly)

    # 7. Persist
    if save:
        save_processed(features)

    # 8. DQ report
    if write_report:
        write_dq_report(dq_report, features)

    log.info("=" * 60)
    log.info("Pipeline complete. Output: %s rows × %s cols",
             f"{len(features):,}", features.shape[1])
    log.info("=" * 60)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the FORESIGHT data pipeline")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help="Directory containing raw CSVs")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip writing processed data to disk")
    args = parser.parse_args()

    run_pipeline(raw_dir=args.raw_dir, save=not args.no_save)
