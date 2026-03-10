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

FEATURE_COLS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14",
    "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28",
    "sales_ewm_28",
    "dow", "month", "is_weekend",
    "Price", "Discount", "Holiday/Promotion",
    "Competitor Pricing", "competitor_price_ratio",
    "Inventory Level", "Units Ordered",
    "days_of_supply", "price_change",
    "sample_weight",
]

CAT_COLS = ["Weather Condition", "Seasonality", "Category", "Region"]

XGB_PARAMS = {
    "n_estimators":     300,
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH  = Path("data/splits/train.parquet")
TEST_PATH   = Path("data/splits/test.parquet")
OUTPUT_DIR  = Path("model/xgboost")
REPORTS_DIR = Path("reports")


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


# ── Feature Preparation ───────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    available = [c for c in FEATURE_COLS if c in df.columns]
    cats      = [c for c in CAT_COLS      if c in df.columns]

    X = df[available + cats].copy()
    y = df[TARGET_COL].copy()

    for col in cats:
        X[col] = X[col].astype("category").cat.codes

    X = X.fillna(0)
    return X, y


# ── Walk-Forward Validation ───────────────────────────────────────────────────
def walk_forward_xgboost(
    train_df:   pd.DataFrame,
    n_splits:   int  = 5,
    val_months: int  = 2,
    gap_days:   int  = 7,
) -> list[dict]:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    train_df           = train_df.copy()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    train_df           = train_df.sort_values(DATE_COL)

    min_date        = train_df[DATE_COL].min()
    max_date        = train_df[DATE_COL].max()
    val_window_days = val_months * 30
    usable_days     = (max_date - min_date).days - (val_window_days + gap_days)
    step_days       = usable_days // n_splits

    fold_results = []

    for fold_idx in range(n_splits):
        train_end  = min_date + pd.Timedelta(days=step_days * (fold_idx + 1))
        val_start  = train_end  + pd.Timedelta(days=gap_days)
        val_end    = val_start  + pd.Timedelta(days=val_window_days)

        if val_end > max_date:
            val_end = max_date
        if val_start >= max_date:
            break

        fold_train = train_df[train_df[DATE_COL] <= train_end]
        fold_val   = train_df[
            (train_df[DATE_COL] > val_start) &
            (train_df[DATE_COL] <= val_end)
        ]

        if len(fold_train) < 100 or len(fold_val) == 0:
            continue

        X_train, y_train = prepare_features(fold_train)
        X_val,   y_val   = prepare_features(fold_val)

        sample_weight = (
            fold_train["sample_weight"].values
            if "sample_weight" in fold_train.columns else None
        )

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            sample_weight = sample_weight,
            eval_set      = [(X_val, y_val)],
            verbose       = False,
        )

        y_pred  = model.predict(X_val).clip(0)
        metrics = compute_metrics(y_val.values, y_pred)

        baseline_metrics = (
            compute_metrics(y_val.values, fold_val["y_pred_baseline"].fillna(0).values)
            if "y_pred_baseline" in fold_val.columns
            else {"MAE": None, "RMSE": None, "MAPE": None}
        )

        fold_result = {
            "fold":          fold_idx + 1,
            "train_start":   str(min_date.date()),
            "train_end":     str(train_end.date()),
            "val_start":     str(val_start.date()),
            "val_end":       str(val_end.date()),
            "n_train":       len(fold_train),
            "n_val":         len(fold_val),
            "xgb_MAE":       metrics["MAE"],
            "xgb_RMSE":      metrics["RMSE"],
            "xgb_MAPE":      metrics["MAPE"],
            "baseline_MAE":  baseline_metrics["MAE"],
            "baseline_RMSE": baseline_metrics["RMSE"],
        }
        fold_results.append(fold_result)

        log.info(
            "Fold %d | Train end: %s (%d rows) | Val MAE: %.4f | Baseline MAE: %s",
            fold_idx + 1, str(train_end.date()), len(fold_train),
            metrics["MAE"], baseline_metrics["MAE"],
        )

    return fold_results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    log.info("=== XGBoost Training Pipeline ===")
    log.info("Train : %s", TRAIN_PATH)
    log.info("Test  : %s", TEST_PATH)

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df  = pd.read_parquet(TEST_PATH)
    log.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))

    # ── Step 1: Walk-forward validation ───────────────────────────────────────
    log.info("Running walk-forward validation...")
    fold_results = walk_forward_xgboost(train_df, n_splits=5)

    fold_df    = pd.DataFrame(fold_results)
    cv_mae     = round(fold_df["xgb_MAE"].mean(),              4)
    cv_rmse    = round(fold_df["xgb_RMSE"].mean(),             4)
    cv_mape    = round(fold_df["xgb_MAPE"].mean(),             4)
    cv_baseline = round(fold_df["baseline_MAE"].dropna().mean(), 4)

    log.info("CV Results → MAE: %.4f | RMSE: %.4f | MAPE: %.4f%%", cv_mae, cv_rmse, cv_mape)

    # ── Step 2: Final model on full train set ──────────────────────────────────
    log.info("Training final model on full train set...")

    X_train, y_train = prepare_features(train_df)
    X_test,  y_test  = prepare_features(test_df)

    sample_weight = (
        train_df["sample_weight"].values
        if "sample_weight" in train_df.columns else None
    )

    final_model = XGBRegressor(**XGB_PARAMS)
    final_model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)

    # ── Step 3: Test set evaluation ────────────────────────────────────────────
    y_pred       = final_model.predict(X_test).clip(0)
    test_metrics = compute_metrics(y_test.values, y_pred)

    baseline_test = (
        compute_metrics(y_test.values, test_df["y_pred_baseline"].fillna(0).values)
        if "y_pred_baseline" in test_df.columns
        else {"MAE": None, "RMSE": None, "MAPE": None}
    )

    # ── Step 4: Feature importance ─────────────────────────────────────────────
    feature_importance = dict(sorted(
        zip(X_train.columns.tolist(), [float(s) for s in final_model.feature_importances_]),
        key=lambda x: x[1], reverse=True
    ))

    log.info("Top 5 features:")
    for feat, score in list(feature_importance.items())[:5]:
        log.info("  %s: %.4f", feat, score)

    # ── Step 5: Improvement calculations ──────────────────────────────────────
    improvement_vs_baseline = (
        round((baseline_test["MAE"] - test_metrics["MAE"]) / baseline_test["MAE"] * 100, 2)
        if baseline_test["MAE"] else None
    )

    # ── Step 6: Save outputs ───────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    agg_metrics = {
        "model":                       "xgboost",
        "n_train":                     len(train_df),
        "n_test":                      len(test_df),
        "cv_mean_MAE":                 cv_mae,
        "cv_mean_RMSE":                cv_rmse,
        "cv_mean_MAPE":                cv_mape,
        "cv_baseline_MAE":             cv_baseline,
        "test_MAE":                    test_metrics["MAE"],
        "test_RMSE":                   test_metrics["RMSE"],
        "test_MAPE":                   test_metrics["MAPE"],
        "baseline_MAE":                baseline_test["MAE"],
        "baseline_RMSE":               baseline_test["RMSE"],
        "improvement_vs_baseline_pct": improvement_vs_baseline,
        "xgb_params":                  XGB_PARAMS,
        "top_features":                {k: float(v) for k, v in list(feature_importance.items())[:10]},
        "trained_at":                  datetime.utcnow().isoformat() + "Z",
        "git_commit":                  os.environ.get("GITHUB_SHA",      "local")[:8],
        "git_branch":                  os.environ.get("GITHUB_REF_NAME", "local"),
        "ci_run_id":                   os.environ.get("GITHUB_RUN_ID",   "local"),
    }

    metrics_path = REPORTS_DIR / "xgboost_metrics.json"
    fold_path    = REPORTS_DIR / "xgboost_folds.json"

    with open(metrics_path, "w") as f:
        json.dump(agg_metrics, f, indent=2)

    with open(fold_path, "w") as f:
        json.dump(fold_results, f, indent=2)

    log.info("Metrics → %s", metrics_path)
    log.info("Folds   → %s", fold_path)

    log.info("=== XGBoost Results ===")
    log.info("  Test MAE  : %.4f", test_metrics["MAE"])
    log.info("  Test RMSE : %.4f", test_metrics["RMSE"])
    log.info("  Test MAPE : %.4f%%", test_metrics["MAPE"])
    log.info("  Baseline MAE           : %s", baseline_test["MAE"])
    log.info("  Improvement vs baseline: %s%%", improvement_vs_baseline)


if __name__ == "__main__":
    main()