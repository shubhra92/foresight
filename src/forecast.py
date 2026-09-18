"""
forecast.py
===========
Project FORESIGHT — Demand Forecasting
Client: NorthBay Living

Responsibilities
----------------
1. Seasonal-Naive baseline  — predict demand = same week last year.
2. LightGBM forecast model  — trained on engineered features from pipeline.py.
3. Rolling-Origin CV        — time-series backtesting with NO data leakage.
4. WAPE evaluation          — primary metric; bias reported as secondary.
5. Prediction intervals     — empirical 80 % interval from residual distribution.
6. Persistence              — save / load trained model artefacts.

Public API
----------
    from src.forecast import run_forecast

    results = run_forecast()
    # results keys: 'forecast_df', 'backtest_metrics', 'model'

Non-negotiable rule (per brief §7)
-----------------------------------
Beat the seasonal-naive baseline honestly.
  * Rolling-origin CV only — never a single random split.
  * No future data ever enters a feature (all lags shift > 0).
  * WAPE reported vs baseline in every backtest result.
  * If LightGBM does not beat baseline → baseline is shipped and reported.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FORECAST_HORIZON_WEEKS: int = 8          # default forward horizon
CV_N_SPLITS: int = 5                     # rolling-origin folds
CV_TEST_SIZE_WEEKS: int = 4              # weeks per fold test window
MIN_HISTORY_WEEKS: int = 8              # minimum history to train on a SKU

# LightGBM feature columns (must all exist in weekly feature frame)
FEATURE_COLS: list[str] = [
    "lag_1w", "lag_2w", "lag_4w",
    "roll_mean_4w", "roll_std_4w", "roll_mean_8w",
    "week_of_year", "month", "is_q4", "season_encoded",
    "price_ratio", "promo_weeks", "is_holiday_week",
    "sku_age_weeks", "stock_cover_weeks",
    "cat_Bedding", "cat_Decor", "cat_Furniture",
    "cat_Kitchen", "cat_Lighting",
]

TARGET_COL: str = "units_sold"
DATE_COL: str = "week_start"
SKU_COL: str = "sku_id"


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error.

    WAPE = Σ|actual - predicted| / Σ|actual|

    Robust to zero-demand SKUs where MAPE blows up.
    Returns a value in [0, ∞); lower is better.
    """
    total_actual = np.abs(actual).sum()
    if total_actual == 0:
        return 0.0
    return float(np.abs(actual - predicted).sum() / total_actual)


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean forecast error (signed).
    Positive → over-forecast; negative → under-forecast.
    """
    if len(actual) == 0:
        return 0.0
    return float((predicted - actual).mean())


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_absolute_error(actual, predicted))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Seasonal-Naive Baseline
# ─────────────────────────────────────────────────────────────────────────────

def seasonal_naive_forecast(
    df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON_WEEKS,
) -> pd.DataFrame:
    """
    Seasonal-Naive baseline: predict demand for week W = demand for week W−52.

    For SKUs with < 52 weeks history, fall back to the per-SKU weekly mean
    of available history (graceful cold-start handling).

    Parameters
    ----------
    df      : Weekly feature frame (output of pipeline.run_pipeline()).
    horizon : Number of future weeks to forecast.

    Returns
    -------
    pd.DataFrame with columns:
        sku_id, week_start, baseline_forecast
    """
    df = df.copy().sort_values([SKU_COL, DATE_COL])
    records: list[dict] = []

    for sku_id, grp in df.groupby(SKU_COL):
        grp = grp.sort_values(DATE_COL).reset_index(drop=True)
        last_date: pd.Timestamp = grp[DATE_COL].max()

        # Historical mean for cold-start fallback
        hist_mean = grp[TARGET_COL].mean()

        for h in range(1, horizon + 1):
            future_date = last_date + pd.Timedelta(weeks=h)
            future_week = future_date.isocalendar().week

            # Find the matching week in prior year
            same_week_prior = grp[
                grp[DATE_COL].dt.isocalendar().week == future_week
            ][TARGET_COL]

            if len(same_week_prior) > 0:
                forecast_val = float(same_week_prior.iloc[-1])  # most recent match
            else:
                forecast_val = float(hist_mean)

            records.append({
                SKU_COL: sku_id,
                DATE_COL: future_date,
                "baseline_forecast": max(forecast_val, 0.0),
            })

    return pd.DataFrame(records)


def seasonal_naive_backtest(
    df: pd.DataFrame,
    n_splits: int = CV_N_SPLITS,
    test_size: int = CV_TEST_SIZE_WEEKS,
) -> dict:
    """
    Rolling-origin evaluation of the seasonal-naive baseline.

    Returns
    -------
    dict with keys: wape, bias, mae, fold_details
    """
    df = df.sort_values([SKU_COL, DATE_COL])
    all_dates = sorted(df[DATE_COL].unique())
    n = len(all_dates)

    fold_wapes, fold_biases, fold_maes = [], [], []
    fold_details = []

    for fold in range(n_splits):
        # Rolling origin: each fold moves the cutoff forward by test_size
        test_end_idx = n - (n_splits - fold - 1) * test_size - 1
        test_start_idx = test_end_idx - test_size + 1
        if test_start_idx <= 52:          # need at least 1 year of history
            continue

        train_dates = all_dates[:test_start_idx]
        test_dates = all_dates[test_start_idx: test_end_idx + 1]

        train_df = df[df[DATE_COL].isin(train_dates)]
        test_df = df[df[DATE_COL].isin(test_dates)]

        # Generate baseline predictions for test window
        preds = []
        for sku_id, grp in train_df.groupby(SKU_COL):
            grp = grp.sort_values(DATE_COL)
            hist_mean = grp[TARGET_COL].mean()
            for test_row in test_df[test_df[SKU_COL] == sku_id].itertuples():
                tw = test_row.week_start.isocalendar().week
                match = grp[grp[DATE_COL].dt.isocalendar().week == tw][TARGET_COL]
                pred = float(match.iloc[-1]) if len(match) > 0 else float(hist_mean)
                preds.append({
                    SKU_COL: sku_id,
                    DATE_COL: test_row.week_start,
                    "actual": test_row.units_sold,
                    "baseline_forecast": max(pred, 0.0),
                })

        if not preds:
            continue

        fold_df = pd.DataFrame(preds)
        actual = fold_df["actual"].values
        predicted = fold_df["baseline_forecast"].values

        fw = wape(actual, predicted)
        fb = bias(actual, predicted)
        fm = mae(actual, predicted)

        fold_wapes.append(fw)
        fold_biases.append(fb)
        fold_maes.append(fm)
        fold_details.append({
            "fold": fold + 1,
            "train_cutoff": str(all_dates[test_start_idx - 1].date()),
            "test_window": f"{test_dates[0].date()} → {test_dates[-1].date()}",
            "wape": round(fw, 4),
            "bias": round(fb, 4),
            "mae": round(fm, 4),
        })

    return {
        "wape": round(float(np.mean(fold_wapes)), 4) if fold_wapes else None,
        "bias": round(float(np.mean(fold_biases)), 4) if fold_biases else None,
        "mae":  round(float(np.mean(fold_maes)), 4) if fold_maes else None,
        "fold_details": fold_details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. LightGBM Forecast Model
# ─────────────────────────────────────────────────────────────────────────────

def _get_lgbm():
    """Lazy-import LightGBM so the module is importable even without it."""
    try:
        import lightgbm as lgb
        return lgb
    except ImportError as exc:
        raise ImportError(
            "LightGBM is not installed. Run: pip install lightgbm"
        ) from exc


def _resolve_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the subset of FEATURE_COLS that actually exist in df.
    Logs a warning for any missing ones.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning("Missing feature columns (skipped): %s", missing)
    return available


def train_lgbm(
    train_df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    seed: int = 42,
) -> object:
    """
    Train a LightGBM regressor on the training slice.

    Uses quantile-safe parameters and median-imputes NaN lags so cold-start
    SKUs can still be scored.

    Parameters
    ----------
    train_df     : Weekly feature frame rows used for training.
    feature_cols : Which columns to use as features.
    seed         : Random seed for reproducibility.

    Returns
    -------
    Trained LightGBM Booster object.
    """
    lgb = _get_lgbm()

    if feature_cols is None:
        feature_cols = _resolve_feature_cols(train_df)

    X = train_df[feature_cols].copy()
    y = train_df[TARGET_COL].values.astype(float)

    # Median-impute NaN lag/rolling features (cold-start rows)
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    params = {
        "objective": "regression_l1",   # MAE objective → robust to outliers
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": seed,
        "n_jobs": -1,
    }

    train_set = lgb.Dataset(X, label=y, free_raw_data=False)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[train_set],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    return model


def predict_lgbm(
    model,
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> np.ndarray:
    """
    Generate point forecasts from a trained LightGBM model.
    NaN features are median-imputed using the prediction frame itself.
    """
    if feature_cols is None:
        feature_cols = _resolve_feature_cols(df)

    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    preds = model.predict(X)
    return np.clip(preds, 0, None)   # demand cannot be negative


# ─────────────────────────────────────────────────────────────────────────────
# 3. Rolling-Origin Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def rolling_origin_cv(
    df: pd.DataFrame,
    n_splits: int = CV_N_SPLITS,
    test_size: int = CV_TEST_SIZE_WEEKS,
    seed: int = 42,
) -> dict:
    """
    Rolling-origin (expanding-window) cross-validation for LightGBM.

    For each fold:
        Train  : all weeks UP TO the cutoff date.
        Test   : the next `test_size` weeks.
    The cutoff advances by `test_size` each fold → no overlap, no leakage.

    Returns
    -------
    dict with keys:
        wape, bias, mae           — mean across folds
        fold_details              — per-fold metrics
        oof_df                    — out-of-fold predictions DataFrame
    """
    df = df.sort_values([SKU_COL, DATE_COL]).copy()
    all_dates = sorted(df[DATE_COL].unique())
    n = len(all_dates)

    fold_wapes, fold_biases, fold_maes = [], [], []
    fold_details = []
    oof_records = []

    feature_cols = _resolve_feature_cols(df)

    for fold in range(n_splits):
        test_end_idx = n - (n_splits - fold - 1) * test_size - 1
        test_start_idx = test_end_idx - test_size + 1

        if test_start_idx < MIN_HISTORY_WEEKS:
            log.warning("Fold %d skipped: insufficient training history", fold + 1)
            continue

        train_dates = all_dates[:test_start_idx]
        test_dates = all_dates[test_start_idx: test_end_idx + 1]

        train_df = df[df[DATE_COL].isin(train_dates)].copy()
        test_df = df[df[DATE_COL].isin(test_dates)].copy()

        if len(train_df) < 50 or len(test_df) == 0:
            continue

        # Train model on this fold
        model = train_lgbm(train_df, feature_cols=feature_cols, seed=seed)
        preds = predict_lgbm(model, test_df, feature_cols=feature_cols)

        actual = test_df[TARGET_COL].values.astype(float)

        fw = wape(actual, preds)
        fb = bias(actual, preds)
        fm = mae(actual, preds)

        fold_wapes.append(fw)
        fold_biases.append(fb)
        fold_maes.append(fm)

        log.info(
            "Fold %d/%d | train→%s | test %s→%s | WAPE=%.4f | bias=%.2f | MAE=%.2f",
            fold + 1, n_splits,
            train_dates[-1].date(),
            test_dates[0].date(), test_dates[-1].date(),
            fw, fb, fm,
        )

        fold_details.append({
            "fold": fold + 1,
            "train_cutoff": str(train_dates[-1].date()),
            "test_window": f"{test_dates[0].date()} → {test_dates[-1].date()}",
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
            "wape": round(fw, 4),
            "bias": round(fb, 4),
            "mae": round(fm, 4),
        })

        oof_df = test_df[[SKU_COL, DATE_COL, TARGET_COL]].copy()
        oof_df["lgbm_forecast"] = preds
        oof_df["fold"] = fold + 1
        oof_records.append(oof_df)

    oof_combined = pd.concat(oof_records, ignore_index=True) if oof_records else pd.DataFrame()

    return {
        "wape": round(float(np.mean(fold_wapes)), 4) if fold_wapes else None,
        "bias": round(float(np.mean(fold_biases)), 4) if fold_biases else None,
        "mae":  round(float(np.mean(fold_maes)), 4) if fold_maes else None,
        "fold_details": fold_details,
        "oof_df": oof_combined,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model selection — baseline vs LightGBM
# ─────────────────────────────────────────────────────────────────────────────

def select_model(
    lgbm_wape: Optional[float],
    baseline_wape: Optional[float],
) -> str:
    """
    Honest model selection per brief §7:
    Ship LightGBM only if it beats the baseline on backtest WAPE.
    Otherwise ship the baseline and report why.
    """
    if lgbm_wape is None or baseline_wape is None:
        log.warning("Insufficient backtest data — defaulting to baseline.")
        return "baseline"

    if lgbm_wape < baseline_wape:
        improvement = (baseline_wape - lgbm_wape) / baseline_wape * 100
        log.info(
            "Model selection: LightGBM wins | WAPE %.4f vs baseline %.4f "
            "(%.1f %% improvement)",
            lgbm_wape, baseline_wape, improvement,
        )
        return "lgbm"
    else:
        log.warning(
            "Model selection: Baseline wins | WAPE lgbm=%.4f vs baseline=%.4f. "
            "Shipping seasonal-naive baseline.",
            lgbm_wape, baseline_wape,
        )
        return "baseline"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Prediction interval (empirical 80 %)
# ─────────────────────────────────────────────────────────────────────────────

def compute_prediction_intervals(
    oof_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    point_col: str = "forecast",
    alpha: float = 0.80,
) -> pd.DataFrame:
    """
    Attach empirical prediction intervals to the forward forecast.

    Method: compute absolute residuals on out-of-fold predictions,
    then use the (1-alpha)/2 and (1+alpha)/2 percentiles as the interval
    half-width around each point forecast.

    This gives a single global interval band; a per-SKU variant would
    require many more OOF samples.

    Parameters
    ----------
    oof_df      : Out-of-fold predictions from rolling_origin_cv().
    forecast_df : Forward forecast DataFrame (sku_id, week_start, forecast).
    point_col   : Name of the forecast column in forecast_df.
    alpha       : Coverage level (default 0.80 → 80 % interval).

    Returns
    -------
    forecast_df with two new columns: forecast_lo, forecast_hi
    """
    if oof_df.empty or "lgbm_forecast" not in oof_df.columns:
        forecast_df["forecast_lo"] = (forecast_df[point_col] * 0.75).clip(0)
        forecast_df["forecast_hi"] = forecast_df[point_col] * 1.25
        return forecast_df

    residuals = (oof_df["lgbm_forecast"] - oof_df[TARGET_COL]).abs()
    lo_q = (1 - alpha) / 2
    hi_q = 1 - lo_q
    lo_delta = float(np.quantile(residuals, lo_q))
    hi_delta = float(np.quantile(residuals, hi_q))

    forecast_df = forecast_df.copy()
    forecast_df["forecast_lo"] = (forecast_df[point_col] - hi_delta).clip(0)
    forecast_df["forecast_hi"] = forecast_df[point_col] + hi_delta

    return forecast_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Forward forecast generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_forward_forecast(
    df: pd.DataFrame,
    model,
    model_type: str,
    horizon: int = FORECAST_HORIZON_WEEKS,
    oof_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generate a forward forecast for all SKUs over `horizon` weeks.

    For LightGBM we roll the forecast horizon one week at a time,
    using lag features from actual history for week 1, then
    recursively from the predicted values for weeks 2+.

    Parameters
    ----------
    df         : Full weekly feature frame (pipeline output).
    model      : Trained LightGBM model (or None for baseline).
    model_type : 'lgbm' or 'baseline'.
    horizon    : Weeks to forecast forward.
    oof_df     : OOF predictions (for interval estimation).

    Returns
    -------
    pd.DataFrame — columns:
        sku_id, week_start, forecast, forecast_lo, forecast_hi,
        model_type, category, list_price
    """
    df = df.sort_values([SKU_COL, DATE_COL]).copy()

    if model_type == "baseline":
        fwd = seasonal_naive_forecast(df, horizon=horizon)
        fwd = fwd.rename(columns={"baseline_forecast": "forecast"})
        fwd["forecast_lo"] = (fwd["forecast"] * 0.75).clip(0)
        fwd["forecast_hi"] = fwd["forecast"] * 1.25
        fwd["model_type"] = "seasonal_naive"
    else:
        feature_cols = _resolve_feature_cols(df)
        records: list[dict] = []

        for sku_id, grp in df.groupby(SKU_COL):
            grp = grp.sort_values(DATE_COL).reset_index(drop=True)
            last_date = grp[DATE_COL].max()

            # Build a rolling buffer of recent actuals for lag construction
            recent_units: list[float] = grp[TARGET_COL].tolist()

            for h in range(1, horizon + 1):
                future_date = last_date + pd.Timedelta(weeks=h)
                last_row = grp.iloc[-1].copy()

                # Construct lag features from the rolling buffer
                row_feats = {}
                buf_len = len(recent_units)
                row_feats["lag_1w"] = recent_units[buf_len - 1] if buf_len >= 1 else np.nan
                row_feats["lag_2w"] = recent_units[buf_len - 2] if buf_len >= 2 else np.nan
                row_feats["lag_4w"] = recent_units[buf_len - 4] if buf_len >= 4 else np.nan

                # Rolling stats from buffer
                row_feats["roll_mean_4w"] = float(np.mean(recent_units[-4:])) if buf_len >= 1 else np.nan
                row_feats["roll_std_4w"]  = float(np.std(recent_units[-4:]))  if buf_len >= 2 else 0.0
                row_feats["roll_mean_8w"] = float(np.mean(recent_units[-8:])) if buf_len >= 1 else np.nan

                # Calendar features for the future date
                row_feats["week_of_year"]   = int(future_date.isocalendar().week)
                row_feats["month"]          = int(future_date.month)
                row_feats["is_q4"]          = int(future_date.month in [10, 11, 12])
                row_feats["season_encoded"] = int(last_row.get("season_encoded", 2))
                row_feats["price_ratio"]    = float(last_row.get("price_ratio", 1.0))
                row_feats["promo_weeks"]    = 0   # conservative: no promo assumed
                row_feats["is_holiday_week"] = 0
                row_feats["sku_age_weeks"]  = int(last_row.get("sku_age_weeks", 0)) + h
                row_feats["stock_cover_weeks"] = float(last_row.get("stock_cover_weeks", 0))

                # Category dummies from last row
                for col in feature_cols:
                    if col.startswith("cat_") and col not in row_feats:
                        row_feats[col] = int(last_row.get(col, 0))

                # Fill any remaining feature cols
                for col in feature_cols:
                    if col not in row_feats:
                        row_feats[col] = float(last_row.get(col, 0) or 0)

                feat_df = pd.DataFrame([row_feats])
                # Ensure correct column order
                feat_df = feat_df.reindex(columns=feature_cols, fill_value=0)

                pred = float(model.predict(feat_df)[0])
                pred = max(pred, 0.0)

                records.append({
                    SKU_COL: sku_id,
                    DATE_COL: future_date,
                    "forecast": pred,
                    "category": grp["category"].iloc[-1],
                    "list_price": float(grp["list_price"].iloc[-1]),
                })

                recent_units.append(pred)  # use predicted value for next lag

        fwd = pd.DataFrame(records)
        fwd["model_type"] = "lgbm"

        if oof_df is not None and not oof_df.empty:
            fwd = compute_prediction_intervals(oof_df, fwd, point_col="forecast")
        else:
            fwd["forecast_lo"] = (fwd["forecast"] * 0.75).clip(0)
            fwd["forecast_hi"] = fwd["forecast"] * 1.25

    # Attach category and list_price if not already present
    meta = df.groupby(SKU_COL)[["category", "list_price"]].last().reset_index()
    if "category" not in fwd.columns:
        fwd = fwd.merge(meta, on=SKU_COL, how="left")
    elif "list_price" not in fwd.columns:
        fwd = fwd.merge(meta[[SKU_COL, "list_price"]], on=SKU_COL, how="left")

    fwd["forecast"]    = fwd["forecast"].round(2)
    fwd["forecast_lo"] = fwd["forecast_lo"].round(2)
    fwd["forecast_hi"] = fwd["forecast_hi"].round(2)

    return fwd.sort_values([SKU_COL, DATE_COL]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Persist / load model artefacts
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, path: Path = MODELS_DIR / "lgbm_model.joblib") -> Path:
    """Serialise model to disk with joblib."""
    joblib.dump(model, path)
    log.info("Model saved → %s", path)
    return path


def load_model(path: Path = MODELS_DIR / "lgbm_model.joblib"):
    """Load a previously saved model artefact."""
    if not path.exists():
        raise FileNotFoundError(f"Model artefact not found: {path}")
    model = joblib.load(path)
    log.info("Model loaded ← %s", path)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_forecast(
    weekly_df: Optional[pd.DataFrame] = None,
    horizon: int = FORECAST_HORIZON_WEEKS,
    save: bool = True,
) -> dict:
    """
    Execute the full FORESIGHT forecasting pipeline.

    Steps
    -----
    1. Load weekly feature frame (from pipeline.py) if not provided.
    2. Baseline backtest  — rolling-origin CV on seasonal-naive.
    3. LightGBM backtest  — rolling-origin CV on LightGBM.
    4. Model selection    — honest comparison on backtest WAPE.
    5. Final model        — retrain winner on full history.
    6. Forward forecast   — generate `horizon` weeks ahead for all SKUs.
    7. Prediction intervals — empirical 80 % band.
    8. Persist            — save model + forecast CSV.

    Parameters
    ----------
    weekly_df : Pre-loaded feature frame. If None, loads from processed/.
    horizon   : Forecast horizon in weeks.
    save      : Persist model and forecast to disk.

    Returns
    -------
    dict with keys:
        'forecast_df'       — forward forecast DataFrame
        'backtest_lgbm'     — LightGBM CV metrics
        'backtest_baseline' — Baseline CV metrics
        'model_type'        — 'lgbm' or 'baseline'
        'model'             — trained model object (or None for baseline)
        'wape_comparison'   — dict summarising the head-to-head
    """
    log.info("=" * 60)
    log.info("PROJECT FORESIGHT — Forecasting pipeline starting")
    log.info("=" * 60)

    # 1. Load data
    if weekly_df is None:
        parquet = PROCESSED_DIR / "weekly_features.parquet"
        csv_path = PROCESSED_DIR / "weekly_features.csv"
        if parquet.exists():
            weekly_df = pd.read_parquet(parquet)
        elif csv_path.exists():
            weekly_df = pd.read_csv(csv_path, parse_dates=["week_start", "launch_date"])
        else:
            raise FileNotFoundError(
                "Processed data not found. Run pipeline.py first."
            )
        log.info("Loaded processed data: %s rows", f"{len(weekly_df):,}")

    weekly_df[DATE_COL] = pd.to_datetime(weekly_df[DATE_COL])

    # 2. Baseline backtest
    log.info("── Seasonal-Naive baseline backtest ──")
    bt_baseline = seasonal_naive_backtest(weekly_df, n_splits=CV_N_SPLITS,
                                          test_size=CV_TEST_SIZE_WEEKS)
    log.info("Baseline WAPE=%.4f | bias=%.2f | MAE=%.2f",
             bt_baseline["wape"] or 0,
             bt_baseline["bias"] or 0,
             bt_baseline["mae"] or 0)

    # 3. LightGBM backtest
    log.info("── LightGBM rolling-origin CV ──")
    bt_lgbm = rolling_origin_cv(weekly_df, n_splits=CV_N_SPLITS,
                                 test_size=CV_TEST_SIZE_WEEKS)
    log.info("LightGBM WAPE=%.4f | bias=%.2f | MAE=%.2f",
             bt_lgbm["wape"] or 0,
             bt_lgbm["bias"] or 0,
             bt_lgbm["mae"] or 0)

    # 4. Model selection
    chosen = select_model(bt_lgbm["wape"], bt_baseline["wape"])

    # 5. Retrain final model on full history
    oof_df = bt_lgbm.get("oof_df", pd.DataFrame())
    if chosen == "lgbm":
        log.info("── Retraining LightGBM on full history ──")
        feature_cols = _resolve_feature_cols(weekly_df)
        final_model = train_lgbm(weekly_df, feature_cols=feature_cols)
    else:
        final_model = None

    # 6 + 7. Forward forecast with intervals
    log.info("── Generating %d-week forward forecast ──", horizon)
    forecast_df = generate_forward_forecast(
        weekly_df,
        model=final_model,
        model_type=chosen,
        horizon=horizon,
        oof_df=oof_df,
    )
    log.info("Forecast rows: %s", f"{len(forecast_df):,}")

    # 8. Persist
    if save:
        if final_model is not None:
            save_model(final_model)
        fcast_path = PROCESSED_DIR / "forecast.parquet"
        forecast_df.to_parquet(fcast_path, index=False, engine="pyarrow")
        forecast_df.to_csv(PROCESSED_DIR / "forecast.csv", index=False)
        log.info("Forecast saved → %s", fcast_path)

    wape_comparison = {
        "lgbm_wape":     bt_lgbm["wape"],
        "baseline_wape": bt_baseline["wape"],
        "winner":        chosen,
        "improvement_pct": (
            round((bt_baseline["wape"] - bt_lgbm["wape"])
                  / bt_baseline["wape"] * 100, 1)
            if bt_lgbm["wape"] and bt_baseline["wape"]
            else None
        ),
    }

    # Persist the backtest evidence so the dashboard and API can report it
    # without re-running the CV (walk-forward roll) each time.
    if save:
        metrics_path = REPORTS_DIR / "backtest_metrics.json"
        backtest_metrics = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "model_type": chosen,
            "horizon_weeks": horizon,
            "cv": {
                "n_splits": CV_N_SPLITS,
                "test_size_weeks": CV_TEST_SIZE_WEEKS,
                "method": "rolling-origin (expanding window), no future data in features",
            },
            "wape_comparison": wape_comparison,
            "lgbm": {
                "wape": bt_lgbm["wape"],
                "bias": bt_lgbm["bias"],
                "mae":  bt_lgbm["mae"],
                "folds": bt_lgbm["fold_details"],
            },
            "baseline": {
                "model": "seasonal_naive",
                "wape": bt_baseline["wape"],
                "bias": bt_baseline["bias"],
                "mae":  bt_baseline["mae"],
                "folds": bt_baseline["fold_details"],
            },
        }
        with open(metrics_path, "w") as fh:
            json.dump(backtest_metrics, fh, indent=2)
        log.info("Backtest metrics saved → %s", metrics_path)

    log.info("=" * 60)
    log.info("Forecasting complete | model=%s | horizon=%d wks", chosen, horizon)
    log.info("WAPE comparison: lgbm=%.4f  baseline=%.4f  winner=%s",
             wape_comparison["lgbm_wape"] or 0,
             wape_comparison["baseline_wape"] or 0,
             wape_comparison["winner"])
    log.info("=" * 60)

    return {
        "forecast_df":       forecast_df,
        "backtest_lgbm":     bt_lgbm,
        "backtest_baseline": bt_baseline,
        "model_type":        chosen,
        "model":             final_model,
        "wape_comparison":   wape_comparison,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run FORESIGHT forecasting")
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON_WEEKS)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_forecast(horizon=args.horizon, save=not args.no_save)

    print("\n── WAPE Comparison ─────────────────────────────────")
    print(json.dumps(results["wape_comparison"], indent=2))
    print("\n── Fold Details (LightGBM) ─────────────────────────")
    for f in results["backtest_lgbm"]["fold_details"]:
        print(f"  Fold {f['fold']} | {f['test_window']} | WAPE={f['wape']}")
    print(f"\n── Forecast sample ─────────────────────────────────")
    print(results["forecast_df"].head(10).to_string(index=False))
