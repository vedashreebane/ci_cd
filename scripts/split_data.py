from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14",
    "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28",
    "sales_ewm_28",
    "dow", "month", "is_weekend",
    "Price", "Discount", "Holiday/Promotion",
    "Competitor Pricing", "competitor_price_ratio",
    "Weather Condition", "Seasonality",
    "Category", "Region",
    "Inventory Level", "Units Ordered",
    "days_of_supply", "price_change",
    "sample_weight",
]

LABEL_COL       = "y"
DATE_COL        = "as_of_date"
IDENTIFIER_COLS = ["Store ID", "Product ID"]

# ── Paths (all relative to repo root) ────────────────────────────────────────
INPUT_PATH  = Path("data/input/features.parquet")
OUTPUT_DIR  = Path("data/splits")
REPORTS_DIR = Path("reports")


# ── Data Classes ──────────────────────────────────────────────────────────────
@dataclass
class SplitResult:
    train:      pd.DataFrame
    val:        pd.DataFrame
    test:       pd.DataFrame
    train_end:  str = ""
    val_end:    str = ""
    test_end:   str = ""
    n_train:    int = 0
    n_val:      int = 0
    n_test:     int = 0

    def summary(self) -> dict:
        total = self.n_train + self.n_val + self.n_test
        return {
            "train_rows": self.n_train,
            "val_rows":   self.n_val,
            "test_rows":  self.n_test,
            "train_end":  self.train_end,
            "val_end":    self.val_end,
            "test_end":   self.test_end,
            "train_pct":  round(self.n_train / total * 100, 1),
            "val_pct":    round(self.n_val   / total * 100, 1),
            "test_pct":   round(self.n_test  / total * 100, 1),
        }


@dataclass
class WalkForwardFold:
    fold_number:  int
    train:        pd.DataFrame
    val:          pd.DataFrame
    train_start:  str = ""
    train_end:    str = ""
    val_start:    str = ""
    val_end:      str = ""
    n_train:      int = 0
    n_val:        int = 0


# ── Core Functions ─────────────────────────────────────────────────────────────
def chronological_split(
    df:         pd.DataFrame,
    train_frac: float = 0.80,
    val_frac:   float = 0.10,
    date_col:   str   = DATE_COL,
) -> SplitResult:
    assert train_frac + val_frac < 1.0, "train_frac + val_frac must be < 1.0"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    dates          = df[date_col].sort_values().unique()
    n_dates        = len(dates)
    train_end_idx  = int(n_dates * train_frac)
    val_end_idx    = int(n_dates * (train_frac + val_frac))

    train_end_date = dates[train_end_idx - 1]
    val_end_date   = dates[val_end_idx - 1]

    train = df[df[date_col] <= train_end_date].copy()
    val   = df[(df[date_col] > train_end_date) & (df[date_col] <= val_end_date)].copy()
    test  = df[df[date_col] > val_end_date].copy()

    result = SplitResult(
        train     = train,
        val       = val,
        test      = test,
        train_end = str(train_end_date.date()),
        val_end   = str(val_end_date.date()),
        test_end  = str(df[date_col].max().date()),
        n_train   = len(train),
        n_val     = len(val),
        n_test    = len(test),
    )

    log.info("=== Chronological Split ===")
    log.info("Train : %d rows  (ends %s)", result.n_train, result.train_end)
    log.info("Val   : %d rows  (ends %s)", result.n_val,   result.val_end)
    log.info("Test  : %d rows  (ends %s)", result.n_test,  result.test_end)
    return result


def walk_forward_validation(
    df:         pd.DataFrame,
    n_splits:   int = 5,
    val_months: int = 2,
    gap_days:   int = 7,
    date_col:   str = DATE_COL,
) -> list[WalkForwardFold]:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    min_date        = df[date_col].min()
    max_date        = df[date_col].max()
    total_days      = (max_date - min_date).days
    val_window_days = val_months * 30
    usable_days     = total_days - (val_window_days + gap_days)
    step_days       = usable_days // n_splits

    folds = []
    for fold_idx in range(n_splits):
        train_end_date = min_date + pd.Timedelta(days=step_days * (fold_idx + 1))
        val_start_date = train_end_date + pd.Timedelta(days=gap_days)
        val_end_date   = val_start_date + pd.Timedelta(days=val_window_days)

        if val_end_date > max_date:
            val_end_date = max_date
        if val_start_date >= max_date:
            break

        train_fold = df[df[date_col] <= train_end_date].copy()
        val_fold   = df[
            (df[date_col] > val_start_date) &
            (df[date_col] <= val_end_date)
        ].copy()

        if len(train_fold) == 0 or len(val_fold) == 0:
            continue

        fold = WalkForwardFold(
            fold_number = fold_idx + 1,
            train       = train_fold,
            val         = val_fold,
            train_start = str(min_date.date()),
            train_end   = str(train_end_date.date()),
            val_start   = str(val_start_date.date()),
            val_end     = str(val_end_date.date()),
            n_train     = len(train_fold),
            n_val       = len(val_fold),
        )
        folds.append(fold)

        log.info(
            "Fold %d | Train: %s → %s (%d rows) | Val: %s → %s (%d rows)",
            fold.fold_number,
            fold.train_start, fold.train_end, fold.n_train,
            fold.val_start,   fold.val_end,   fold.n_val,
        )

    return folds


def get_X_y(
    df:           pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    label_col:    str       = LABEL_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.warning("Missing feature columns (will be skipped): %s", missing)

    X = df[available].copy()
    y = df[label_col].copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype("category").cat.codes

    return X, y


def save_splits(split: SplitResult, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fixed names — no run_id suffix needed since GCS folders are versioned by run_id
    paths = {
        "train": str(output_dir / "train.parquet"),
        "val":   str(output_dir / "val.parquet"),
        "test":  str(output_dir / "test.parquet"),
    }

    split.train.to_parquet(paths["train"], index=False)
    split.val.to_parquet(paths["val"],     index=False)
    split.test.to_parquet(paths["test"],   index=False)

    log.info("train.parquet → %s  (%d rows)", paths["train"], len(split.train))
    log.info("val.parquet   → %s  (%d rows)", paths["val"],   len(split.val))
    log.info("test.parquet  → %s  (%d rows)", paths["test"],  len(split.test))
    return paths


def save_report(summary: dict, folds: list, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = {
        **summary,
        "n_folds": len(folds),
        "folds": [
            {
                "fold":        f.fold_number,
                "train_start": f.train_start,
                "train_end":   f.train_end,
                "val_start":   f.val_start,
                "val_end":     f.val_end,
                "n_train":     f.n_train,
                "n_val":       f.n_val,
            }
            for f in folds
        ],
        # CI context — auto-set by GitHub Actions
        "git_commit":   os.environ.get("GITHUB_SHA",      "local")[:8],
        "git_branch":   os.environ.get("GITHUB_REF_NAME", "local"),
        "ci_run_id":    os.environ.get("GITHUB_RUN_ID",   "local"),
        "triggered_by": os.environ.get("GITHUB_ACTOR",    "local"),
        "timestamp":    datetime.utcnow().isoformat() + "Z",
    }

    report_path = reports_dir / "split_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("Split report saved to %s", report_path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=== Data Split Pipeline ===")
    log.info("Input  : %s", INPUT_PATH)
    log.info("Output : %s", OUTPUT_DIR)

    # 1. Load
    df = pd.read_parquet(INPUT_PATH)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # 2. Chronological split
    split = chronological_split(df, train_frac=0.80, val_frac=0.10)

    # 3. Walk-forward folds (on train set only — for cross-validation during training)
    folds = walk_forward_validation(split.train, n_splits=5)

    # 4. Save parquet files
    save_splits(split, OUTPUT_DIR)

    # 5. Save summary report (uploaded as GitHub artifact)
    save_report(split.summary(), folds, REPORTS_DIR)

    log.info("=== Done ===")


if __name__ == "__main__":
    main()