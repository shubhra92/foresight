"""
generate_data.py
================
Synthetic data generator for Project FORESIGHT.

Produces four CSV extracts that mimic what a real D2C brand (NorthBay Living)
would export from their warehouse / ERP system:

    data/raw/sales_daily.csv
    data/raw/sku_master.csv
    data/raw/calendar.csv
    data/raw/inventory_snapshots.csv

Design decisions
----------------
* ~200 active SKUs across 5 categories / 15 subcategories.
* 2 years of daily sales history (730 rows per SKU at most).
* Realistic demand: seasonal pattern + trend + promo lift + noise.
* Deliberate imperfections per the brief:
    - ~3 % missing values scattered across sales_daily (units_sold, revenue).
    - ~1 % duplicate rows in sales_daily.
    - Inconsistent category labels (mixed case, trailing spaces).
    - A handful of SKUs with very sparse history (cold-start scenario).
    - A few negative unit_price values (data-quality issue to catch).
* Random seed fixed so results are reproducible.

Usage
-----
    python data/generate_data.py            # writes to data/raw/
    python data/generate_data.py --rows 300 # smaller smoke-test dataset
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants / configuration
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

N_SKUS = 200
HISTORY_START = "2023-01-01"
HISTORY_END = "2024-12-31"

CATEGORIES: dict[str, list[str]] = {
    "Furniture": ["Sofas", "Beds", "Dining Tables", "Storage"],
    "Decor": ["Wall Art", "Vases", "Cushions", "Rugs"],
    "Lighting": ["Ceiling Lights", "Table Lamps", "Floor Lamps"],
    "Kitchen": ["Cookware", "Tableware", "Small Appliances"],
    "Bedding": ["Bed Sheets", "Pillows", "Duvets"],
}

# Promo event names used in calendar
PROMO_EVENTS = [
    "Diwali Sale",
    "New Year Sale",
    "Summer Clearance",
    "Republic Day",
    "Independence Day",
    "Black Friday",
    "Christmas Sale",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. sku_master
# ─────────────────────────────────────────────────────────────────────────────

def build_sku_master() -> pd.DataFrame:
    """One row per SKU: category, subcategory, launch date, unit cost, list price."""

    cats = list(CATEGORIES.keys())
    rows = []
    launch_dates = pd.date_range(HISTORY_START, periods=N_SKUS, freq="3D")

    for i in range(N_SKUS):
        cat = cats[i % len(cats)]
        subcat = CATEGORIES[cat][i % len(CATEGORIES[cat])]
        launch = launch_dates[i].date()

        # A few cold-start SKUs launched in the last 60 days of history
        if i >= N_SKUS - 10:
            launch = pd.Timestamp(HISTORY_END).date() - pd.Timedelta(days=random.randint(20, 60))

        unit_cost = round(rng.uniform(300, 8000), 2)
        list_price = round(unit_cost * rng.uniform(1.4, 2.8), 2)

        # Inject inconsistent labels for ~10 % of rows (mixed case, trailing spaces)
        if i % 10 == 0:
            cat = cat.upper()
        elif i % 17 == 0:
            cat = cat.lower() + " "

        rows.append(
            {
                "sku_id": f"SKU{i+1:04d}",
                "category": cat,
                "subcategory": subcat,
                "launch_date": launch,
                "unit_cost": unit_cost,
                "list_price": list_price,
            }
        )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. calendar
# ─────────────────────────────────────────────────────────────────────────────

def build_calendar() -> pd.DataFrame:
    """One row per calendar date with seasonality / holiday / promo metadata."""

    dates = pd.date_range(HISTORY_START, HISTORY_END, freq="D")
    df = pd.DataFrame({"date": dates})

    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon

    # Season mapping (Northern Hemisphere proxy for India)
    season_map = {
        1: "Winter", 2: "Winter", 3: "Spring",
        4: "Spring", 5: "Summer", 6: "Summer",
        7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Autumn", 11: "Autumn", 12: "Winter",
    }
    df["season"] = df["month"].map(season_map)

    # Public holidays (India-ish calendar — simplified)
    holidays = {
        "2023-01-26", "2023-03-08", "2023-04-14", "2023-08-15",
        "2023-10-02", "2023-10-24", "2023-11-12", "2023-12-25",
        "2024-01-26", "2024-03-25", "2024-04-14", "2024-08-15",
        "2024-10-02", "2024-10-31", "2024-11-01", "2024-12-25",
    }
    df["is_holiday"] = df["date"].dt.strftime("%Y-%m-%d").isin(holidays).astype(int)

    # Named promo events (multi-day windows)
    promo_windows: list[tuple[str, str, str]] = [
        ("2023-10-20", "2023-10-26", "Diwali Sale"),
        ("2023-12-26", "2024-01-01", "New Year Sale"),
        ("2023-06-01", "2023-06-07", "Summer Clearance"),
        ("2024-01-24", "2024-01-26", "Republic Day"),
        ("2024-08-13", "2024-08-15", "Independence Day"),
        ("2024-11-29", "2024-12-01", "Black Friday"),
        ("2024-12-23", "2024-12-25", "Christmas Sale"),
    ]
    df["promo_event"] = None
    for start, end, name in promo_windows:
        mask = (df["date"] >= start) & (df["date"] <= end)
        df.loc[mask, "promo_event"] = name

    df["date"] = df["date"].dt.date
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. sales_daily
# ─────────────────────────────────────────────────────────────────────────────

def _seasonal_index(date: pd.Timestamp) -> float:
    """Simple seasonal multiplier based on month (peak Oct–Dec)."""
    month_idx = [0.75, 0.78, 0.85, 0.90, 0.92, 0.88,
                 0.80, 0.83, 0.88, 1.10, 1.20, 1.30]
    return month_idx[date.month - 1]


def _trend_factor(date: pd.Timestamp, launch: pd.Timestamp) -> float:
    """Slow upward trend from launch date; growth tapering after 1 year."""
    days_alive = max((date - launch).days, 0)
    return 1.0 + min(days_alive / 730, 0.5)  # max +50 % over 2 years


def build_sales_daily(sku_master: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """One row per SKU per day: units sold, revenue, price, promo flag."""

    dates = pd.date_range(HISTORY_START, HISTORY_END, freq="D")
    promo_dates = set(
        calendar.loc[calendar["promo_event"].notna(), "date"].astype(str)
    )

    rows = []
    for _, sku in sku_master.iterrows():
        sku_id: str = sku["sku_id"]
        launch = pd.Timestamp(sku["launch_date"])
        list_price: float = float(sku["list_price"])

        # Base weekly demand ~ Poisson with category-level mean
        base_demand = rng.integers(3, 40)

        for date in dates:
            if date < launch:
                continue  # SKU not yet live

            seasonal = _seasonal_index(date)
            trend = _trend_factor(date, launch)
            is_promo = date.strftime("%Y-%m-%d") in promo_dates
            promo_lift = rng.uniform(1.3, 2.5) if is_promo else 1.0

            mu = base_demand * seasonal * trend * promo_lift
            units = int(rng.poisson(max(mu, 0.5)))

            # Weekend lift (+15 %)
            if date.dayofweek >= 5:
                units = int(units * rng.uniform(1.1, 1.2))

            # Price: small daily jitter ± 5 %, occasionally a promo discount
            if is_promo:
                unit_price = round(list_price * rng.uniform(0.75, 0.90), 2)
            else:
                unit_price = round(list_price * rng.uniform(0.97, 1.03), 2)

            revenue = round(units * unit_price, 2)

            rows.append(
                {
                    "date": date.date(),
                    "sku_id": sku_id,
                    "units_sold": units,
                    "revenue": revenue,
                    "unit_price": unit_price,
                    "promo_flag": int(is_promo),
                }
            )

    df = pd.DataFrame(rows)

    # ── Deliberate imperfections ──────────────────────────────────────────────

    # 1. ~3 % missing values in units_sold and revenue
    n_missing = int(len(df) * 0.03)
    miss_idx_units = rng.choice(df.index, size=n_missing // 2, replace=False)
    miss_idx_rev = rng.choice(df.index, size=n_missing // 2, replace=False)
    df.loc[miss_idx_units, "units_sold"] = np.nan
    df.loc[miss_idx_rev, "revenue"] = np.nan

    # 2. ~1 % duplicate rows
    n_dupes = int(len(df) * 0.01)
    dupe_idx = rng.choice(df.index, size=n_dupes, replace=False)
    dupes = df.loc[dupe_idx].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    # 3. A few negative unit_price values (data-quality issue)
    neg_idx = rng.choice(df.index, size=10, replace=False)
    df.loc[neg_idx, "unit_price"] = df.loc[neg_idx, "unit_price"] * -1

    return df.sort_values(["date", "sku_id"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. inventory_snapshots
# ─────────────────────────────────────────────────────────────────────────────

def build_inventory_snapshots(sku_master: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly inventory snapshot per SKU.
    Columns: date, sku_id, on_hand_units, on_order_units, lead_time_days,
             reorder_point.
    """

    # One snapshot per week (Monday)
    snapshot_dates = pd.date_range(HISTORY_START, HISTORY_END, freq="W-MON")
    rows = []

    for _, sku in sku_master.iterrows():
        sku_id: str = sku["sku_id"]
        launch = pd.Timestamp(sku["launch_date"])
        lead_time = int(rng.integers(7, 28))   # 1–4 weeks
        reorder_pt = int(rng.integers(10, 80))

        on_hand = int(rng.integers(50, 500))   # starting stock

        for snap_date in snapshot_dates:
            if snap_date < launch:
                continue

            # Simple random walk to simulate stock movement week-over-week
            weekly_sales_approx = int(rng.integers(5, 50))
            on_hand = max(0, on_hand - weekly_sales_approx)

            # Trigger a replenishment order when near reorder point
            on_order = 0
            if on_hand <= reorder_pt:
                on_order = int(rng.integers(100, 400))
                on_hand += on_order  # order arrives next week (simplified)
                on_order = 0         # received

            rows.append(
                {
                    "date": snap_date.date(),
                    "sku_id": sku_id,
                    "on_hand_units": on_hand,
                    "on_order_units": on_order,
                    "lead_time_days": lead_time,
                    "reorder_point": reorder_pt,
                }
            )

    return pd.DataFrame(rows).sort_values(["date", "sku_id"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(n_skus: int = N_SKUS) -> None:
    global N_SKUS
    N_SKUS = n_skus

    print(f"[generate_data] Building synthetic dataset — {N_SKUS} SKUs, "
          f"{HISTORY_START} → {HISTORY_END}")

    print("  Building sku_master …")
    sku_master = build_sku_master()
    sku_master.to_csv(RAW_DIR / "sku_master.csv", index=False)
    print(f"    sku_master     : {len(sku_master):>6,} rows → {RAW_DIR / 'sku_master.csv'}")

    print("  Building calendar …")
    calendar = build_calendar()
    calendar.to_csv(RAW_DIR / "calendar.csv", index=False)
    print(f"    calendar       : {len(calendar):>6,} rows → {RAW_DIR / 'calendar.csv'}")

    print("  Building sales_daily … (this takes a moment)")
    sales = build_sales_daily(sku_master, calendar)
    sales.to_csv(RAW_DIR / "sales_daily.csv", index=False)
    print(f"    sales_daily    : {len(sales):>6,} rows → {RAW_DIR / 'sales_daily.csv'}")

    print("  Building inventory_snapshots …")
    inventory = build_inventory_snapshots(sku_master)
    inventory.to_csv(RAW_DIR / "inventory_snapshots.csv", index=False)
    print(f"    inventory_snapshots: {len(inventory):>6,} rows → {RAW_DIR / 'inventory_snapshots.csv'}")

    print("[generate_data] Done. All four extracts written to data/raw/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FORESIGHT synthetic data")
    parser.add_argument(
        "--skus", type=int, default=N_SKUS,
        help="Number of SKUs to generate (default: 200)"
    )
    args = parser.parse_args()
    main(n_skus=args.skus)
