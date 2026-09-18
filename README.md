# 📦 Project FORESIGHT — Demand & Inventory Intelligence

**Client:** NorthBay Living &nbsp;|&nbsp; **Platform:** Zidio Development &nbsp;|&nbsp; **Role:** Data Scientist

> _Turn NorthBay's raw sales and inventory data into a demand forecast and an early-warning system that tells the planning team what to reorder, what to clear, and what to leave alone._

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [Source Data Model](#3-source-data-model)
4. [Project Structure](#4-project-structure)
5. [Setup & Installation](#5-setup--installation)
6. [Run the Pipeline (end-to-end)](#6-run-the-pipeline-end-to-end)
7. [Backtest Results — WAPE vs Baseline](#7-backtest-results--wape-vs-baseline)
8. [Dashboard](#8-dashboard)
9. [Scoring Service (API)](#9-scoring-service-api)
10. [Stakeholders & RACI](#10-stakeholders--raci)
11. [Key Assumptions & Limitations](#11-key-assumptions--limitations)
12. [Deliverable Checklist](#12-deliverable-checklist)
13. [Submission Requirements](#13-submission-requirements)
14. [Deployment, Tests & Operations](#14-deployment-tests--operations)

---

## 1. Problem Statement

NorthBay Living (~200 active SKUs, online-only D2C brand) plans inventory on gut feel and spreadsheets. Two costly failure modes recur every month:

| Failure | Impact |
|---|---|
| **Stockout** — best-sellers run out | Lost revenue, frustrated customers |
| **Overstock** — slow movers pile up | Cash locked in stock, margin-eroding markdowns |

**Brief from Head of Operations:**
> _"I need something that tells me, for each product, how much we'll likely sell over the next few weeks, which products are about to run out, and which ones are overstocked so we can clear them — and it has to be something my team can actually look at."_

---

## 2. Solution Overview

```
Raw CSVs  →  Pipeline  →  Forecast Model  →  Risk Scoring  →  Dashboard + API
```

| Component | What it does |
|---|---|
| `src/pipeline.py` | Ingests, cleans, merges, and feature-engineers the 4 raw tables into a weekly SKU-level feature frame |
| `src/forecast.py` | Seasonal-naive baseline + LightGBM model; rolling-origin CV; honest WAPE comparison |
| `src/risk.py` | Stockout / overstock scoring [0,1]; 4-quadrant decisioning; rupee impact quantification |
| `app/app.py` | Streamlit planning dashboard (KPIs, forecast chart, decisioning grid, action tables) |
| `service/main.py` | FastAPI scoring service — `/predict`, `/summary`, `/skus`, `/health` |

---

## 3. Source Data Model

Four raw tables (per the client engagement brief) feed the pipeline. Latest data is generated with `data/generate_data.py`; real NorthBay exports drop into `data/raw/` and re-run clean.

| Table | Grain | Key columns |
|---|---|---|
| `sku_master` | 1 row per SKU | `sku_id`, category, subcategory, list price, unit cost, launch date |
| `calendar` | 1 row per day | date, week, month, season, holiday flag |
| `sales_daily` (transactions) | 1 row per SKU per day | `sku_id`, units sold, revenue, avg unit price, promo flag |
| `inventory_snapshots` | 1 row per SKU per day | `sku_id`, on-hand, on-order, lead time, reorder point |

The pipeline aggregates these into a weekly SKU-level feature frame (`weekly_features.parquet` — one row per SKU per week), which every downstream step reads.

---

## 4. Project Structure

```
foresight/
├── data/
│   ├── raw/                    # sales_daily, sku_master, calendar, inventory_snapshots
│   └── processed/              # weekly_features, forecast, risk_scores (parquet + csv)
├── notebooks/                  # EDA and experimentation
├── src/
│   ├── pipeline.py             # ingest → clean → merge → aggregate → feature-engineer
│   ├── forecast.py             # baseline + LightGBM + rolling-origin CV + intervals
│   └── risk.py                 # stockout/overstock scoring + rupee impact
├── app/
│   └── app.py                  # Streamlit dashboard
├── service/
│   └── main.py                 # FastAPI scoring endpoint
├── models/                     # serialised model artefacts (joblib)
├── reports/
│   ├── data_quality_report.txt # auto-generated DQ report
│   ├── backtest_metrics.json   # live WAPE vs baseline evidence (used by dashboard + API)
│   └── executive_readout.md    # D7 — 1-page executive readout
├── README.md
└── requirements.txt
```

---

## 5. Setup & Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone / navigate to the repo
cd foresight

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Run the Pipeline (end-to-end)

Run each step in order. Every step is independent and idempotent.

```bash
# Step 1 — Generate synthetic data (skip if real CSVs are in data/raw/)
python data/generate_data.py

# Step 2 — Data pipeline: clean, merge, feature-engineer
python src/pipeline.py

# Step 3 — Forecasting: train model, run rolling-origin CV, generate forecast
python src/forecast.py

# Step 4 — Risk scoring: quadrant classification + rupee impact
python src/risk.py

# Step 5 — Launch the planning dashboard
streamlit run app/app.py

# Step 6 (optional) — Start the scoring API
uvicorn service.main:app --host 0.0.0.0 --port 8000 --reload
```

**One-liner for steps 1–4:**
```bash
python data/generate_data.py && python src/pipeline.py && python src/forecast.py && python src/risk.py
```

---

## 7. Backtest Results — WAPE vs Baseline

> Results are from rolling-origin cross-validation (5 folds, 4-week test windows).  
> No future data was used in any feature. See `src/forecast.py` for full methodology.  
> Live copy: `reports/backtest_metrics.json` (surfaced in the dashboard and via `GET /evaluation`).

| Model | WAPE | Bias | MAE |
|---|---|---|---|
| **Seasonal-Naive Baseline** | 0.2618 | -43.17 | 61.46 |
| **LightGBM** | **0.1203** | 10.89 | 27.68 |
| **Winner** | ✅ LightGBM — **54.0 % improvement** | — | — |

Rolling-origin CV (5 folds, 4-week test windows each):

| Fold | Test Window | LightGBM WAPE |
|---|---|---|
| 1 | 2024-08-19 → 2024-09-09 | 0.1065 |
| 2 | 2024-09-16 → 2024-10-07 | 0.0819 |
| 3 | 2024-10-14 → 2024-11-04 | 0.0857 |
| 4 | 2024-11-11 → 2024-12-02 | 0.0907 |
| 5 | 2024-12-09 → 2024-12-30 | 0.2367 |

> The model selection is honest: if LightGBM does not beat the baseline on backtest WAPE, the baseline is shipped and reported (not hidden).

### Forecast accuracy metric — WAPE

```
WAPE = Σ|actual − predicted| / Σ|actual|
```

WAPE is used as the primary metric because it is robust to zero-demand SKUs (where MAPE explodes). Bias (mean signed error) is reported as a secondary metric to catch systematic over/under-forecasting.

---

## 8. Dashboard

```bash
streamlit run app/app.py
```

Opens at `http://localhost:8501`.

**Views:**
- **Planning Dashboard** — KPI cards (Reorder count, Markdown count, Sales at risk ₹, Locked capital ₹), a **forecast-accuracy panel (WAPE vs baseline + per-fold bar chart)**, inventory risk matrix (stockout × overstock, bubble = ₹ at stake), priority-action table, and an actual-vs-forecast revenue outlook with 80 % confidence interval.
- **Reorder Plan** — sorted replenishment table with suggested PO qty, unit cost, PO value, lead time and status pills.
- **Stockout Alerts** — early-warning table (risk score, days of cover, weekly velocity, sales at risk, demand trend) with category breakdown and risk thresholds.
- **Markdown Candidates** — overstocked SKUs with suggested discount, new price, excess units and capital locked.

**Sidebar:** API URL + live-service status badge, filters (Category, Risk Quadrant), model info, and a per-SKU drill-down (live forecast via the API when online, cached forecast otherwise).

---

## 9. Scoring Service (API)

```bash
uvicorn service.main:app --host 0.0.0.0 --port 8000
```

Interactive docs: `http://localhost:8000/docs`

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health check + data-loaded status |
| `GET` | `/skus` | List all SKU IDs with category and current quadrant |
| `GET` | `/summary` | Portfolio-level KPIs (quadrant counts, ₹ at stake) |
| `GET` | `/evaluation` | Rolling-origin backtest evidence — WAPE vs baseline, per-fold detail |
| `POST` | `/predict` | Forecast + risk for one or more SKUs |

### Example — single SKU prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sku_ids": ["SKU0001"], "horizon_weeks": 8}'
```

**Response shape:**
```json
{
  "results": [
    {
      "sku_id": "SKU0001",
      "category": "Furniture",
      "stockout_risk": 0.72,
      "overstock_risk": 0.18,
      "quadrant": "REORDER NOW",
      "recommended_action": "Raise a replenishment order before stock runs out.",
      "sales_at_risk_inr": 84320.00,
      "locked_capital_inr": 0.00,
      "rupee_at_stake": 84320.00,
      "suggested_reorder_qty": 124,
      "on_hand_units": 45,
      "avg_weekly_demand": 18.5,
      "forecast": [
        {"week_start": "2025-01-06", "forecast": 19.2, "forecast_lo": 11.4, "forecast_hi": 27.0},
        ...
      ],
      "model_type": "lgbm"
    }
  ],
  "not_found": []
}
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sku_ids": ["SKU0001", "SKU0042", "SKU0100"], "horizon_weeks": 4}'
```

### Error handling

| Scenario | Behaviour |
|---|---|
| Unknown SKU ID | Listed in `not_found`; does not crash the request |
| `horizon_weeks` < 1 or > 26 | 422 Unprocessable Entity with clear message |
| Data not yet generated | 503 Service Unavailable with instructions |
| Empty `sku_ids` list | 422 Unprocessable Entity |

---

## 10. Stakeholders & RACI

| Role | R | A | C | I |
|---|---|---|---|---|
| **Head of Operations — NorthBay** (client) | — | ✅ Approve readouts, sign off actions | — | Informed of risk posture |
| **Data Scientist — you (this engagement)** | ✅ Build pipeline, model, risk, dashboard | ✅ Own analysis quality | Consult on approach | — |
| **Zidio Development (program sponsor)** | — | ✅ Accept deliverables | ✅ Provide brief, data, review | Oversee progress |
| **Planning / Ops team (end users)** | ✅ Run reorder & markdown lists | — | ✅ Set lead times, inventory norms | Consulted via weekly checkpoints |

**Weekly cadence (per brief):** each milestone ends at a client checkpoint where the deliverable is presented, not just handed over.

---

## 11. Key Assumptions & Limitations

| Assumption | Detail |
|---|---|
| Data representativeness | Synthetic extracts mimic real NorthBay behaviour; conclusions transfer to real data after re-running the pipeline |
| History sufficiency | Most SKUs have ≥ 52 weeks of history; 10 cold-start SKUs fall back to category-level patterns |
| Lead times | Lead times and reorder points in inventory data are treated as broadly accurate |
| Promo forecast | Forward forecast uses promo_weeks = 0 (conservative); real promos should be injected via the feature frame |
| Inventory snapshots | Weekly snapshots are forward-filled to daily grain; intra-week movements are not captured |

**Known limitations:**
- `lag_52w` is NULL for SKUs with < 1 year of history — handled by median imputation at scoring time.
- The 80 % prediction interval is a global empirical band, not per-SKU calibrated.
- Risk thresholds (2-week stockout, 8-week overstock) are configurable in `src/risk.py` but not yet exposed via the dashboard UI.

---

## 12. Deliverable Checklist

| # | Deliverable | Status |
|---|---|---|
| D1 | Reproducible data pipeline (`src/pipeline.py`) | ✅ |
| D2 | Data-quality & EDA insight memo (`reports/data_quality_report.txt`) | ✅ |
| D3 | Demand forecast model — backtested, beats baseline (`src/forecast.py`) | ✅ |
| D4 | Risk scoring with recommended actions (`src/risk.py`) | ✅ |
| D5 | Planning dashboard (`app/app.py`) | ✅ |
| D6 | Deployed scoring service (`service/main.py`) | ✅ |
| D7 | Executive readout (`reports/executive_readout.md`) | ✅ |

---

## 13. Submission Requirements

Per the client engagement brief — this repo satisfies all of them:

1. **Repository** — public/private access with the pipeline, notebooks and clean code (`README.md` included).
2. **Live dashboard** — `streamlit run app/app.py` (Planning, Reorder, Stockout, Markdown views).
3. **Live scoring service** — `uvicorn service.main:app --port 8000` with `/health`, `/predict`, `/summary`, `/evaluation`, `/skus`.
4. **The pitch** — business problem, setup steps, backtest evidence (**WAPE vs baseline**), and key assumptions are documented in this README and `reports/executive_readout.md`.
5. **Executive readout** — 1-page for the Head of Operations in `reports/executive_readout.md`.

> **Confidentiality:** the simulated-but-realistic data is treated as confidential project data — do not post it publicly or share duplicates.

---

## 14. Deployment, Tests & Operations

> **Client & ops walkthrough:** see [`docs/ONBOARDING.md`](docs/ONBOARDING.md)
> for the plain-language client onboarding (what data to hand over, weekly
> routine) and the engineer runbook (refresh, containers, troubleshooting).

### Test suite

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q                                          # unit + functional + dashboard
```

### Docker (recommended)

```bash
cp .env.example .env                               # add FORESIGHT_AUTH_TOKEN if needed
docker compose up --build -d                       # builds image, starts API + dashboard
docker compose ps                                  # both healthchecks should be green
```

| Service | Port | Healthcheck |
|---------|------|-------------|
| API | 8000 | `GET /health` |
| Dashboard | 8501 | `_stcore/health` |

**Appearance.** FORESIGHT ships dark-by-default (matches the Figma design).
Switch between light and dark whenever you like with the **☀️ Light / 🌙 Dark**
toggle in the sidebar — the whole dashboard (cards, tables, charts) restyles
instantly, and the choice is kept across browser reloads via the URL. Theme
tokens are pinned in `.streamlit/config.toml` (accent `#F59E0B`), which is
copied into the Docker image, so both deployed and local runs look identical.

### Scheduled weekly refresh (accuracy monitoring)

```bash
python src/run_weekly.py                           # recompute forecast + append accuracy snapshot
# or in Docker:
docker compose --profile refresh run --rm refresh
```

History is written to `reports/accuracy_tracking.parquet`; the dashboard
plots WAPE-over-time automatically and `/evaluation` surfaces `history`
and `last_refresh`.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FORESIGHT_API_URL` | `http://localhost:8000` | Dashboard → API base URL |
| `FORESIGHT_DATA_DIR` | `data/processed` | Where to read parquet artefacts |
| `FORESIGHT_AUTH_TOKEN` | *(empty/open)* | Bearer token for `/predict`, `/skus`, `/summary`, `/evaluation` |
| `FORESIGHT_CORS_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Comma-separated CORS origins |
| `FORESIGHT_STOCKOUT_THRESHOLD_WEEKS` | `2.0` | Risk threshold (`src/risk.py`) |
| `FORESIGHT_OVERSTOCK_THRESHOLD_WEEKS` | `8.0` | Risk threshold |
| `FORESIGHT_RISK_HIGH_THRESHOLD` | `0.5` | Risk threshold |
| `REBUILD` | `0` | `1` → full pipeline rebuild on next container boot |

### CI (GitHub Actions)

`.github/workflows/ci.yml` runs syntax checks, pytest, AppTest, and builds the Docker image on every push/PR.

---

_Project FORESIGHT v1.0 · Zidio Development · Deliver it like a consultant. Defend it like a scientist._
