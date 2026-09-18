# Project FORESIGHT — Onboarding & Operations Runbook

Two audiences, one document:

| Part | Audience | Purpose |
|------|----------|---------|
| [Part 1 — Client onboarding](#part-1--client-onboarding--getting-your-real-data-in) | NorthBay planning / ops team | Plain-language: what to hand over, what you get back |
| [Part 2 — Zidio ops runbook](#part-2--zidio-ops-runbook) | Zidio engineering | Repeatable refreshes, container ops, troubleshooting, accuracy monitoring |

> Everything described here is automated end-to-end. The team does **not**
> train models, tune parameters, or decide weights — the app only ships a
> model that beat its baseline in backtesting.

---

## Part 1 — Client onboarding: getting your real data in

### 1.1 The six-step starting checklist

| # | Step | Who | Details |
|---|------|-----|---------|
| 1 | Export **4 CSV files** from your ERP/OMS/warehouse | Data/ops analyst | See §1.2 for exact files and columns |
| 2 | Send them to Zidio, or drop them into `data/raw/` | Both | Replace the demo files — names must match exactly |
| 3 | Zidio runs the pipeline once | Zidio | `python src/pipeline.py && python src/forecast.py && python src/risk.py` |
| 4 | Check the **Data Quality Report** | Zidio → client sign-off | `reports/data_quality_report.txt` — see §1.3 |
| 5 | Record the first accuracy snapshot | Zidio | `python src/run_weekly.py` |
| 6 | Open the dashboard and confirm the numbers | Client | See §1.4 for what you should see |

### 1.2 The four files the pipeline needs

| File | What it is | How often it changes | Critical columns |
|------|-----------|-----------------------|------------------|
| `sku_master.csv` | Product catalogue | Only when products change | `sku_id`, `category`, `subcategory`, `unit_cost`, `list_price`, `launch_date` |
| `sales_daily.csv` | Every sale, one row per SKU per day | **Weekly** (append new days) | `date`, `sku_id`, `units_sold`, `revenue`, `unit_price`, `promo_flag` |
| `inventory_snapshots.csv` | On-hand stock position, one row per SKU per **week** | **Weekly** (ops snapshot) | `date`, `sku_id`, `on_hand_units`, `on_order_units`, `lead_time_days`, `reorder_point` |
| `calendar.csv` | Retail calendar (season/holiday/promo) | Only when the calendar changes | `date`, `week`, `season`, `is_holiday`, `promo_event` |

Rules of thumb:

- **History:** `sales_daily` must cover at least ~52 weeks of history so the
  year-ago (seasonality) feature works. If some SKUs are younger, they are
  cold-started on category patterns automatically — nothing breaks.
- **Format:** dates `YYYY-MM-DD`, prices/quantities numeric, `sku_id` a string
  like `SKU0001`. The pipeline validates all of this before a single forecast
  is produced.
- **Cadence:** exports don't need to be real-time. A **weekly** export is the
  intended rhythm (the model re-runs weekly anyway).

### 1.3 The Data Quality gate

The first pipeline run produces `reports/data_quality_report.txt`. It
flags a small, fixed set of things — this is your "should we trust this?"
signal:

| Check | What it means |
|-------|---------------|
| Duplicate rows | Same sale recorded twice |
| Negative `unit_price` / `units_sold` | Sign-error exports |
| Orphan SKUs in sales/inventory | SKU ids not present in `sku_master` |
| Null counts | Empty cells that will be imputed |
| Date range | How much history you actually gave us |

Cleaning is automatic and documented per rule (e.g. negative prices are
absolutised, missing revenue is recomputed as units × price). Anything
auto-fixed is logged — you always know what was assumed.

### 1.4 What you get back (first successful login)

- **Planning Dashboard** — portfolio KPIs, 8-week revenue outlook with a
  range, decisioning matrix, and the **Forecast Accuracy** panel (does the
  live model beat the naive baseline? currently yes: WAPE ≈ 12% vs ≈ 26%).
- **Reorder Plan** — what to buy, how much (`suggested_reorder_qty`), PO value
  implied by cost × qty, sorted by urgency.
- **Stockout Alerts** — SKUs that will run out inside supplier lead time
  (cover < 2 weeks), with days left and sales at risk.
- **Markdown Candidates** — overstocked SKUs (cover > 8 weeks), excess units,
  suggested discount and new price, capital locked.

Every tab exports to CSV — hand the export straight to purchasing.

> **Look & feel.** The dashboard opens in dark mode. Prefer light? Use the
> **☀️ Light / 🌙 Dark** toggle at the top of the left sidebar — everything
> restyles on the spot, and your choice is remembered across reloads.

### 1.5 Weekly routine (your side, ~15 minutes per week)

1. **Monday** — ops takes the weekly stock snapshot.
2. Overwrite two files in `data/raw/`: `inventory_snapshots.csv` (new week's
   rows) and `sales_daily.csv` (append the new week's sales).
3. Tell Zidio "data is in" — or, if Zidio has set you up with self-service:
   - Host cron already installed → **nothing to do**, it runs Mondays 03:00.
   - Docker one-shot → `docker compose --profile refresh run --rm refresh`.
4. Open the dashboard and work the four tabs. Numbers are fresh, exports in
   one click.

**You never:** choose model settings, set weights, decide forecast length, or
re-run anything by hand. The only deliberate choice in the whole system is the
risk threshold (2-week stockout / 8-week overstock) — and that's a business
decision the client can tune without code (`FORESIGHT_STOCKOUT_THRESHOLD_WEEKS`,
`FORESIGHT_OVERSTOCK_THRESHOLD_WEEKS` in `.env`).

---

## Part 2 — Zidio ops runbook

### 2.1 Environment

| Thing | Value |
|-------|-------|
| Python | 3.10+ (3.13 used in dev/CI/Docker) |
| Venv | `foresight/.venv` (contains pyarrow, lightgbm, etc.) |
| Data | `data/raw/` (client exports) → `data/processed/` (parquet artefacts) |
| Reports | `reports/` — DQ report, backtest metrics, exec readout, accuracy tracking |
| Model | `models/lgbm_model.joblib` |

Pipeline order (always this sequence):
`src/pipeline.py → src/forecast.py → src/risk.py` then optionally
`src/run_weekly.py` to record accuracy.

### 2.2 Onboarding a new client dataset

```bash
# 1. Drop the four CSVs (exact names) into data/raw/
# 2. Rebuild everything
./.venv/bin/python src/pipeline.py && \
  ./.venv/bin/python src/forecast.py && \
  ./.venv/bin/python src/risk.py
# 3. Read the DQ gate before trusting outputs
open reports/data_quality_report.txt
# 4. Seed the accuracy history (must have SCP'd or run above)
./.venv/bin/python src/run_weekly.py
# 5. Verified start
./.venv/bin/python -m uvicorn service.main:app --host 127.0.0.1 --port 8000
./.venv/bin/python -m streamlit run app/app.py
```

Or in Docker (build once, primed volumes):

```bash
cp .env.example .env          # set FORESIGHT_AUTH_TOKEN to lock the API down
REBUILD=1 docker compose up -d --build
```

`REBUILD=1` makes `run.sh` re-run the pipeline/forecast/risk inside the
container on boot; without it, containers boot straight from the parquet
artefacts baked into the image/named volumes.

### 2.3 Weekly refresh (the only recurring task)

Preferred: host cron (already shipped as `cron.example`, Mondays 03:00 →
`./.venv/bin/python src/run_weekly.py`). It:

1. re-runs pipeline + forecast + risk,
2. appends a WAPE/model snapshot to `reports/accuracy_tracking.parquet`,
3. writes `reports/last_refresh.json`,
4. is visible to the dashboard (drift chart) and the API (`/health` →
   `last_refresh`, `/evaluation` → `history`).

Docker equivalent: `docker compose --profile refresh run --rm refresh`.
If you run the containers for long stretches, restart the API after a refresh
so the in-memory store picks up new artefacts:
`docker compose restart api`.

### 2.4 Security posture

| Concern | Setting |
|---------|---------|
| API auth | `FORESIGHT_AUTH_TOKEN` — when set, `/predict`, `/skus`, `/summary`, `/evaluation` require `Authorization: Bearer <token>` (`hmac.compare_digest`). `/health` stays open for healthchecks. |
| CORS | `FORESIGHT_CORS_ORIGINS` — default localhost:8501 only; change to the real dashboard origin when deploying publicly. |
| Data dir | `FORESIGHT_DATA_DIR` — point the service at any host-mounted processed dir. |
| Risk thresholds | `FORESIGHT_STOCKOUT_THRESHOLD_WEEKS` / `FORESIGHT_OVERSTOCK_THRESHOLD_WEEKS` / `FORESIGHT_RISK_HIGH_THRESHOLD` — tunable at runtime, no code. |

### 2.5 Troubleshooting

| Symptom | Likely cause → fix |
|---------|--------------------|
| Dashboard "No processed data found" | `_check_data()` failed: the three parquet files aren't in `data/processed/`. Run the pipeline. |
| `/health` returns `503` / `data_loaded=false` | Data failed to load at startup; `error` field shows why (usually a missing file). Run pipeline, restart API. |
| `/predict` → `401` | Token required but not sent. `Authorization: Bearer <token>`. |
| Accuracy panel blank | `reports/backtest_metrics.json` missing → `python src/forecast.py`. |
| Drift chart empty | `reports/accuracy_tracking.parquet` absent → run `src/run_weekly.py` once. |
| Orphan SKUs in DQ report | Sales/inventory reference SKUs not in `sku_master` — reconcile with client before trusting forecasts. |
| Old numbers after refresh | In-memory store loaded at startup — restart the API container (or run `docker compose up -d` to recreate). |

### 2.6 Release checklist

- [ ] `./.venv/bin/python -m pytest -q` → 39 passed, 3 skipped (auth tests run when `FORESIGHT_AUTH_TOKEN` is set)
- [ ] `docker compose config -q` → no YAML errors
- [ ] `docker compose up -d --build` → both services `healthy`
- [ ] `curl -s http://localhost:8000/health` → `"status":"ok"`, `last_refresh` present
- [ ] With token set: `/predict` without token → 401; with token → 200 (8-week forecast)
- [ ] `reports/accuracy_tracking.parquet` has ≥ 1 row after onboarding
- [ ] `data/raw/` files match the contract in `data/README.md`