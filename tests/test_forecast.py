"""Unit tests for src/forecast.py metric helpers and model-selection logic."""

import numpy as np

from forecast import (
    wape,
    bias,
    mae,
    select_model,
    compute_prediction_intervals,
    seasonal_naive_forecast,
)
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Metric math
# ─────────────────────────────────────────────────────────────────────────────

def test_wape_perfect_prediction_is_zero():
    a = np.array([10.0, 0.0, 5.0])
    assert wape(a, a) == 0.0


def test_wape_known_case():
    actual    = np.array([100.0, 200.0])
    predicted = np.array([80.0,  250.0])
    # |100-80| + |200-250| = 20 + 50 = 70; Σ|actual| = 300
    expected = 70 / 300
    assert abs(wape(actual, predicted) - expected) < 1e-9


def test_wape_zero_actual_returns_zero():
    assert wape(np.zeros(3), np.ones(3)) == 0.0


def test_bias_positive_is_over_forecast():
    actual    = np.array([10.0, 10.0])
    predicted = np.array([12.0, 14.0])
    assert bias(actual, predicted) > 0


def test_bias_negative_is_under_forecast():
    assert bias(np.array([10.0]), np.array([5.0])) < 0


def test_mae_perfect_is_zero():
    a = np.array([1.0, 2.0, 3.0])
    assert mae(a, a) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Model selection (honest per brief §7)
# ─────────────────────────────────────────────────────────────────────────────

def test_select_model_lgbm_wins_when_lower_wape():
    assert select_model(lgbm_wape=0.10, baseline_wape=0.20) == "lgbm"


def test_select_model_baseline_wins_when_lgbm_not_better():
    assert select_model(lgbm_wape=0.30, baseline_wape=0.20) == "baseline"


def test_select_model_baseline_wins_on_equal_wape():
    assert select_model(lgbm_wape=0.20, baseline_wape=0.20) == "baseline"


def test_select_model_defaults_to_baseline_when_missing():
    assert select_model(None, 0.20) == "baseline"
    assert select_model(0.10, None) == "baseline"
    assert select_model(None, None) == "baseline"


# ─────────────────────────────────────────────────────────────────────────────
# Prediction intervals
# ─────────────────────────────────────────────────────────────────────────────

def test_intervals_fallback_when_oof_empty():
    fc = pd.DataFrame({"forecast": [10.0, 20.0]})
    out = compute_prediction_intervals(pd.DataFrame(), fc, point_col="forecast")
    assert "forecast_lo" in out.columns
    assert "forecast_hi" in out.columns
    # Fallback: lo = 0.75 * forecast; hi = 1.25 * forecast
    assert out["forecast_lo"].iloc[0] == 10.0 * 0.75
    assert out["forecast_hi"].iloc[1] == 20.0 * 1.25


def test_intervals_lo_is_lower_hi_is_higher():
    oof = pd.DataFrame({
        "units_sold": [10, 12, 8, 11],
        "lgbm_forecast": [10, 11, 9, 13],
    })
    fc = pd.DataFrame({"forecast": [15.0]})
    out = compute_prediction_intervals(oof, fc, point_col="forecast")
    assert out["forecast_lo"].iloc[0] < 15.0
    assert out["forecast_hi"].iloc[0] > 15.0