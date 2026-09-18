"""Unit tests for src/risk.py — scoring, thresholds, quadrant logic."""

import math
import pandas as pd
import pytest

import risk as r


def _inventory(on_hand=0.0, on_order=0.0, demand=1.0, price=4500.0, cost=2000.0):
    """One-row inventory + demand frames for compute_risk_scores."""
    inv = pd.DataFrame([{
        "sku_id": "TEST01",
        "on_hand_units": on_hand,
        "on_order_units": on_order,
        "lead_time_days": 14,
        "reorder_point": 10,
        "list_price": price,
        "unit_cost": cost,
    }])
    dem = pd.DataFrame([{
        "sku_id": "TEST01",
        "avg_weekly_demand": demand,
    }])
    return r.compute_risk_scores(inv, dem)


# ─────────────────────────────────────────────────────────────────────────────
# Cover-weeks normalisation
# ─────────────────────────────────────────────────────────────────────────────

def test_cover_weeks_low_cover_means_high_stockout():
    df = _inventory(on_hand=0, on_order=0, demand=20.0)
    assert df["stockout_risk"].iloc[0] == 1.0        # zero cover → worst stockout
    assert df["overstock_risk"].iloc[0] == 0.0


def test_cover_weeks_high_cover_means_high_overstock():
    df = _inventory(on_hand=800, on_order=0, demand=1.0)
    # 800 weeks cover saturates at MAX_COVER_WEEKS (16) → score 1.0
    assert df["overstock_risk"].iloc[0] == 1.0
    assert df["stockout_risk"].iloc[0] == 0.0


def test_cover_weeks_mid_cover_scores_between():
    df = _inventory(on_hand=8, on_order=0, demand=1.0)
    # 8 weeks ≈ half of MAX_COVER_WEEKS
    assert df["stockout_risk"].iloc[0] == pytest.approx(0.5)
    assert df["overstock_risk"].iloc[0] == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Quadrant assignment
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "so,ov,expected",
    [
        (0.9, 0.2, "REORDER NOW"),
        (0.2, 0.9, "MARKDOWN / CLEAR"),
        (0.9, 0.9, "WATCH / VOLATILE"),
        (0.2, 0.2, "HEALTHY"),
    ],
)
def test_assign_quadrant(so, ov, expected):
    assert r.assign_quadrant(so, ov) == expected


@pytest.mark.parametrize(
    "cover_weeks,expected",
    [
        (0.0, 1.0),
        (4.0, 0.75),   # 1 - 4/16
        (8.0, 0.5),
        (16.0, 0.0),   # saturated at MAX_COVER_WEEKS
    ],
)
def test_normalise_cover_stockout(cover_weeks, expected):
    assert r._normalise_cover(cover_weeks, "stockout") == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────────────────────
# Rupee impact (non-negative, finite)
# ─────────────────────────────────────────────────────────────────────────────

def test_rupee_at_stake_non_negative():
    scored = pd.DataFrame([{
        "sku_id": "TEST01",
        "category": "Furniture",
        "subcategory": "Sofas",
        "stockout_risk": 0.8,
        "overstock_risk": 0.1,
        "on_hand_units": 10,
        "projected_stock": 40,
        "demand_over_lead": 100,
        "demand_over_horizon": 400,
        "list_price": 4500.0,
        "unit_cost": 2000.0,
    }])
    out = r.compute_rupee_impact(scored)
    assert len(out) == 1
    assert math.isfinite(out.iloc[0]["rupee_at_stake"])
    assert out.iloc[0]["rupee_at_stake"] >= 0
    # shortfall = 100 - 40 = 60  → sales at risk = 60 × 4500
    assert out.iloc[0]["sales_at_risk_inr"] == 60 * 4500.0
    # excess = on_hand - demand_over_horizon = 0 (clipped) → no locked capital
    assert out.iloc[0]["locked_capital_inr"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# summarise_risk aggregation
# ─────────────────────────────────────────────────────────────────────────────

def test_summarise_risk_totals_add_up():
    df = pd.DataFrame([
        {"rupee_at_stake": 100.0, "sales_at_risk_inr": 80.0, "locked_capital_inr": 20.0, "quadrant": "REORDER NOW"},
        {"rupee_at_stake": 300.0, "sales_at_risk_inr": 0.0, "locked_capital_inr": 300.0, "quadrant": "MARKDOWN / CLEAR"},
    ])
    s = r.summarise_risk(df)
    assert s["total_sales_at_risk_inr"] == 80.0
    assert s["total_locked_capital_inr"] == 320.0
    assert s["reorder_count"] == 1
    assert s["markdown_count"] == 1
    assert s["total_skus"] == 2