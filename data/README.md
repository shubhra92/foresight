# FORESIGHT — Data Reference

Two layers:

| Layer | Path | Contents | Written by |
|-------|------|----------|------------|
| Raw | `data/raw/` | Client exports, CSVs, one table per file | Export from ERP/CRM |
| Processed | `data/processed/` | Cleaned, engineered parquet artefacts | `src/pipeline.py`, `src/forecast.py`, `src/risk.py` |

## Raw tables (real-format source)

Files must keep these exact names and columns (the pipeline's `load_raw_tables`
in `src/pipeline.py` validates them and writes issues to `reports/dq_report.md`).

### `sku_master.csv` — product catalogue
| Column | Type | Notes |
|--------|------|-------|
| `sku_id` | str | Primary key, e.g. `SKU0001` |
| `category` | str | e.g. `FURNITURE`, `Decor`, `Lighting`, `Kitchen`, `Bedding` |
| `subcategory` | str | e.g. `Sofas`, `Pillows`, `Floor Lamps` |
| `launch_date` | date `YYYY-MM-DD` | First day the SKU could sell |
| `unit_cost` | float | INR, cost of goods |
| `list_price` | float | INR, selling price |

### `sales_daily.csv` — daily transactions (fact)
| Column | Type | Notes |
|--------|------|-------|
| `date` | date | Implied by calendar (2023-01-01 → 2024-week-52) |
| `sku_id` | str | FK → `sku_master` |
| `units_sold` | float | Daily demand units |
| `revenue` | float | INR |
| `unit_price` | float | INR |
| `promo_flag` | int | 0/1 — promotional day |

### `inventory_snapshots.csv` — on-hand positions
| Column | Type | Notes |
|--------|------|-------|
| `date` | date | Snapshot date (weekly cadence) |
| `sku_id` | str | FK → `sku_master` |
| `on_hand_units` | int | Physical stock at snapshot |
| `on_order_units` | int | Open POs (used in net available) |
| `lead_time_days` | int | Supplier lead time in days |
| `reorder_point` | int | Trigger threshold used by ops |

### `calendar.csv` — retail calendar (dimension)
| Column | Type | Notes |
|--------|------|-------|
| `date` | date | One row per day |
| `week` | int | ISO-ish week number |
| `month` / `year` | int | |
| `day_of_week` | int | 0 = Monday |
| `season` | str | `Winter`, `Spring`, `Summer`, `Fall` |
| `is_holiday` | int | 0/1 |
| `promo_event` | str | Empty or event name |

## Processed artefacts

| File | Contents |
|------|----------|
| `weekly_features.parquet` | SKU-week grain, features + target (👉 pipeline) |
| `forecast.parquet` | Forward forecast (future weeks, `forecast`/`forecast_lo`/`forecast_hi`) |
| `risk_scores.parquet` | Per-SKU risk scores, quadrant, rupee impact, reorder qty |

## Bringing in the client's real data

> Start with [`docs/ONBOARDING.md`](../docs/ONBOARDING.md) — it walks the client
> through exactly which exports to produce and the Zidio team through the
> refresh cadence.

1. Replace the four CSVs inside `data/raw/` (keep column names above; the
   filter for SKUs with a full year of history — `MIN_HISTORY_WEEKS` in
   `src/pipeline.py` — trims infant SKUs automatically).
2. Rebuild: `python src/pipeline.py && python src/forecast.py && python src/risk.py`.
3. Restart the API/dashboard, or in Docker:
   `REBUILD=1 docker compose up -d --build`.
4. Record the new accuracy snapshot: `python src/run_weekly.py` (appends to
   `reports/accuracy_tracking.parquet`).

## Demo data

`data/generate_data.py` simulates realistic NorthBay Living rows (shape only —
it is not used by any pipeline code and can be deleted for production).