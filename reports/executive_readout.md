# Project FORESIGHT — Executive Readout
**Client:** NorthBay Living · **Prepared by:** Data Science & Analytics (Zidio Development) · **v1.0**

---

## What we built
An early-warning system that turns NorthBay's sales and inventory data into one question per SKU:
**Reorder now, clear it, or leave it alone?**

| Deliverable | Where |
|---|---|
| Reproducible data pipeline | `src/pipeline.py` |
| Demand forecast model (backtested) | `src/forecast.py` |
| Stockout / overstock risk scoring | `src/risk.py` |
| Planning dashboard (4 views) | `app/app.py` · live at `localhost:8501` |
| Scoring service (API) | `service/main.py` · `localhost:8000` |

---

## Forecast accuracy — the model is honest
Compared on a **walk-forward backtest** (5 rolling-origin folds, 4-week test windows, no future data in any feature):

| | WAPE | Bias | MAE |
|---|---|---|---|
| **LightGBM (shipped)** | **12.0 %** | +10.9 | 27.7 units |
| Seasonal-naive baseline | 26.2 % | −43.2 | 61.5 units |
| **Improvement** | **54.0 % lower error** | less bias | 55% lower |

WAPE = weighted absolute percentage error, robust where slow movers would break MAPE. LightGBM **earned** the right to ship; if it had lost, the baseline would have shipped instead.

---

## Current demand picture (8-week horizon: Jan 6 – Feb 24, 2025)
- **Forecast portfolio revenue:** ≈ **₹179 Cr** over the next 8 weeks.
- **200 SKUs tracked** across Furniture, Decor, Lighting, Kitchen and Bedding (~62 weeks of history each).

## Risk posture — what needs attention
| Signal | Count | Impact |
|---|---|---|
| **REORDER NOW** (stockout risk) | **195 / 200 SKUs** | **₹39.3 Cr** sales at risk if not replenished |
| **MARKDOWN / CLEAR** (overstock) | 5 SKUs | ₹15.3 L locked · **376 excess units** |

**Top 5 — highest ₹ at stake, all critical cover:**
1. SKU0080 · Pillows (Bedding) — ₹1.15 Cr
2. SKU0010 · Bed Sheets (Bedding) — ₹1.14 Cr
3. SKU0023 · Table Lamps (Lighting) — ₹1.13 Cr
4. SKU0090 · Duvets (Bedding) — ₹1.05 Cr
5. SKU0027 · Cushions (Decor) — ₹0.99 Cr

---

## Recommended next steps
1. **Replenish the 195 reorder SKUs this week** — suggested order quantities and PO values are in the «Reorder Plan» tab.
2. **Clear the 5 slow movers** — markdown suggestions (10–25%) are in the «Markdown Candidates» tab; ~₹15 L recoverable.
3. **Adopt the accuracy cadence** — re-run `src/forecast.py` weekly; WAPE is tracked in the dashboard so drift is caught early (holdout has shipping safety stock accordingly).
4. **Feed in the real promo calendar** — the forward forecast is intentionally conservative (`promo_weeks = 0`); the pipeline already accepts promo flags.

---

*Data in this engagement is simulated but realistic and treated as **confidential** — not for external publication. Full reproducibility: `src/forecast.py` regenerates the backtest numbers above on demand.*