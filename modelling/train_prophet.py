from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DATE_COL        = "as_of_date"
TARGET_COL      = "y"
IDENTIFIER_COLS = ["Store ID", "Product ID"]

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH  = Path("data/splits/train.parquet")
TEST_PATH   = Path("data/splits/test.parquet")
OUTPUT_DIR  = Path("model/prophet")
REPORTS_DIR = Path("reports")


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mask = y_true != 0
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if mask.sum() > 0 else float("nan")
    )

    return {
        "MAE":  round(float(mae),  4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
    }


# ── Prophet Training (single series) ─────────────────────────────────────────
def train_prophet_for_series(
    train_df:                pd.DataFrame,
    test_df:                 pd.DataFrame,
    store_id:                str,
    product_id:              str,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
) -> dict | None:
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Prophet not installed. Run: pip install prophet")

    train_series = train_df[
        (train_df["Store ID"] == store_id) &
        (train_df["Product ID"] == product_id)
    ].copy()

    test_series = test_df[
        (test_df["Store ID"] == store_id) &
        (test_df["Product ID"] == product_id)
    ].copy()

    if len(train_series) < 30:
        log.warning("Series %s_%s has only %d training rows — skipping",
                    store_id, product_id, len(train_series))
        return None

    if len(test_series) == 0:
        log.warning("Series %s_%s has no test rows — skipping", store_id, product_id)
        return None

    # ── Prepare Prophet format ────────────────────────────────────────────────
    prophet_train = pd.DataFrame({
        "ds": pd.to_datetime(train_series[DATE_COL]),
        "y":  train_series[TARGET_COL].values,
    }).dropna()

    has_holiday = "Holiday/Promotion" in train_series.columns
    if has_holiday:
        prophet_train["holiday_promo"] = train_series["Holiday/Promotion"].values

    # ── Train ─────────────────────────────────────────────────────────────────
    model = Prophet(
        changepoint_prior_scale = changepoint_prior_scale,
        seasonality_prior_scale = seasonality_prior_scale,
        yearly_seasonality      = True,
        weekly_seasonality      = True,
        daily_seasonality       = False,
        seasonality_mode        = "multiplicative",
    )

    if has_holiday:
        model.add_regressor("holiday_promo")

    model.fit(prophet_train)

    # ── Forecast ──────────────────────────────────────────────────────────────
    future = pd.DataFrame({"ds": pd.to_datetime(test_series[DATE_COL])})
    if has_holiday:
        future["holiday_promo"] = test_series["Holiday/Promotion"].values

    forecast         = model.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)

    y_true = test_series[TARGET_COL].values
    y_pred = forecast["yhat"].values[:len(y_true)]

    metrics = compute_metrics(pd.Series(y_true), pd.Series(y_pred))

    baseline_metrics = (
        compute_metrics(pd.Series(y_true), test_series["y_pred_baseline"].fillna(0))
        if "y_pred_baseline" in test_series.columns
        else {"MAE": None, "RMSE": None, "MAPE": None}
    )

    return {
        "store_id":        store_id,
        "product_id":      product_id,
        "series_id":       f"{store_id}_{product_id}",
        "n_train":         len(train_series),
        "n_test":          len(test_series),
        "prophet_MAE":     metrics["MAE"],
        "prophet_RMSE":    metrics["RMSE"],
        "prophet_MAPE":    metrics["MAPE"],
        "baseline_MAE":    baseline_metrics["MAE"],
        "baseline_RMSE":   baseline_metrics["RMSE"],
        "baseline_MAPE":   baseline_metrics["MAPE"],
        "forecast_dates":  forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
        "forecast_values": forecast["yhat"].round(2).tolist(),
        "actual_values":   y_true.tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=== Prophet Training Pipeline ===")
    log.info("Train : %s", TRAIN_PATH)
    log.info("Test  : %s", TEST_PATH)

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df  = pd.read_parquet(TEST_PATH)
    log.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))

    # ── Get unique series ─────────────────────────────────────────────────────
    series = train_df[IDENTIFIER_COLS].drop_duplicates().values.tolist()
    log.info("Training Prophet for %d Store × Product series...", len(series))

    # ── Train per series ──────────────────────────────────────────────────────
    results = []
    failed  = 0

    for i, (store_id, product_id) in enumerate(series):
        if i % 10 == 0:
            log.info("Progress: %d / %d series", i, len(series))

        result = train_prophet_for_series(
            train_df   = train_df,
            test_df    = test_df,
            store_id   = store_id,
            product_id = product_id,
        )

        if result:
            results.append(result)
        else:
            failed += 1

    log.info("Completed: %d trained, %d skipped", len(results), failed)

    if not results:
        raise ValueError("No series were successfully trained!")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    results_df  = pd.DataFrame(results)
    agg_metrics = {
        "model":            "prophet",
        "n_series":         len(results),
        "n_failed":         failed,
        "mean_MAE":         round(results_df["prophet_MAE"].mean(), 4),
        "mean_RMSE":        round(results_df["prophet_RMSE"].mean(), 4),
        "mean_MAPE":        round(results_df["prophet_MAPE"].mean(), 4),
        "baseline_MAE":     round(results_df["baseline_MAE"].dropna().mean(), 4),
        "baseline_RMSE":    round(results_df["baseline_RMSE"].dropna().mean(), 4),
        "trained_at":       datetime.utcnow().isoformat() + "Z",
        "git_commit":       os.environ.get("GITHUB_SHA",      "local")[:8],
        "git_branch":       os.environ.get("GITHUB_REF_NAME", "local"),
        "ci_run_id":        os.environ.get("GITHUB_RUN_ID",   "local"),
    }

    if agg_metrics["baseline_MAE"]:
        improvement = round(
            (agg_metrics["baseline_MAE"] - agg_metrics["mean_MAE"])
            / agg_metrics["baseline_MAE"] * 100, 2
        )
        agg_metrics["improvement_vs_baseline_pct"] = improvement
        log.info("Prophet MAE improvement over baseline: %.2f%%", improvement)

    # ── Save outputs ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    forecast_path = OUTPUT_DIR / "prophet_forecasts.parquet"
    metrics_path  = REPORTS_DIR / "prophet_metrics.json"

    results_df.to_parquet(forecast_path, index=False)
    with open(metrics_path, "w") as f:
        json.dump(agg_metrics, f, indent=2)

    log.info("Forecasts → %s", forecast_path)
    log.info("Metrics   → %s", metrics_path)

    log.info("=== Prophet Results ===")
    log.info("  Mean MAE  : %.4f", agg_metrics["mean_MAE"])
    log.info("  Mean RMSE : %.4f", agg_metrics["mean_RMSE"])
    log.info("  Mean MAPE : %.4f%%", agg_metrics["mean_MAPE"])
    log.info("  Baseline MAE : %.4f", agg_metrics["baseline_MAE"])
    log.info("  Improvement  : %.2f%%", agg_metrics.get("improvement_vs_baseline_pct", 0))


if __name__ == "__main__":
    main()