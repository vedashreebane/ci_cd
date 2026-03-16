"""
prepare_local_data.py
=====================
One-shot script for LOCAL MODEL TESTING only.
Replicates what the Airflow DAG does (transform + split) so you can run
the model scripts without Docker / Airflow / MongoDB.

What it does
------------
  1. Reads the two raw CSVs
  2. Runs the full feature engineering pipeline (mirrors data_pipeline.py v2.0)
  3. Saves  data/features.parquet
  4. Splits 80 / 10 / 10 chronologically
  5. Saves  data/splits/train.parquet
             data/splits/val.parquet
             data/splits/test.parquet

Usage
-----
  1. Place this file in the same folder as your CSVs:
       better_retail_store_inventory.csv
       better_inventory_snapshot.csv

  2. Run once:
       python prepare_local_data.py

  3. Then run any model:
       python prophet_model.py
       python xgboost_model.py
       python lstm_model.py
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths — update if your CSVs live elsewhere ────────────────────────────────
RETAIL_CSV   = Path("better_retail_store_inventory.csv")
SNAPSHOT_CSV = Path("better_inventory_snapshot.csv")
DATA_DIR     = Path("data")
SPLITS_DIR   = DATA_DIR / "splits"

HORIZON = 1   # forecast horizon in days


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Feature engineering  (mirrors transform() in data_pipeline.py v2.0)
# ─────────────────────────────────────────────────────────────────────────────
def build_features(retail_path: Path, snapshot_path: Path) -> pd.DataFrame:
    log.info("Loading CSVs ...")
    retail = pd.read_csv(retail_path)
    snap   = pd.read_csv(snapshot_path)
    log.info("Retail: %d rows   Snapshot: %d rows", len(retail), len(snap))

    # ── Quality checks ────────────────────────────────────────────────────────
    before = len(retail)
    retail = retail.drop_duplicates(subset=["Date", "Store ID", "Product ID"])
    if before - len(retail):
        log.warning("Removed %d duplicate rows", before - len(retail))

    for col in ["Units Sold", "Inventory Level"]:
        neg = (retail[col] < 0).sum()
        if neg:
            log.warning("Clipping %d negatives in %s", neg, col)
            retail[col] = retail[col].clip(lower=0)

    # ── Parse & sort ──────────────────────────────────────────────────────────
    retail["Date"] = pd.to_datetime(retail["Date"])
    retail = retail.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

    # ── Snapshot join ─────────────────────────────────────────────────────────
    df = retail.merge(
        snap[["Store ID", "Product ID", "Lead Time Days"]],
        on=["Store ID", "Product ID"],
        how="left",
    )
    missing_lead = df["Lead Time Days"].isnull().sum()
    if missing_lead:
        log.warning("Filling %d missing Lead Time Days with median", missing_lead)
        df["Lead Time Days"] = df["Lead Time Days"].fillna(df["Lead Time Days"].median())

    # ── Lag & rolling features ────────────────────────────────────────────────
    log.info("Engineering features ...")
    gs = df.groupby(["Store ID", "Product ID"])["Units Sold"]

    df["sales_lag_1"]  = gs.shift(1)
    df["sales_lag_7"]  = gs.shift(7)
    df["sales_lag_14"] = gs.shift(14)
    df["sales_lag_28"] = gs.shift(28)

    df["sales_roll_mean_7"]  = gs.transform(lambda x: x.rolling(7,  min_periods=3).mean().shift(1))
    df["sales_roll_mean_14"] = gs.transform(lambda x: x.rolling(14, min_periods=7).mean().shift(1))
    df["sales_roll_mean_28"] = gs.transform(lambda x: x.rolling(28, min_periods=7).mean().shift(1))
    df["sales_roll_std_7"]   = gs.transform(lambda x: x.rolling(7,  min_periods=3).std().shift(1))
    df["sales_ewm_28"]       = gs.transform(lambda x: x.shift(1).ewm(span=28, adjust=False).mean())

    # ── Demand forecast (lagged to prevent leakage) ───────────────────────────
    df["demand_forecast_lag1"] = df.groupby(
        ["Store ID", "Product ID"]
    )["Demand Forecast"].shift(1)

    # ── Pricing ───────────────────────────────────────────────────────────────
    df["price_vs_competitor"] = df["Price"] / df["Competitor Pricing"].clip(lower=0.01)
    df["effective_price"]     = df["Price"] * (1 - df["Discount"] / 100)

    # ── Inventory position ────────────────────────────────────────────────────
    df["stockout_flag"]    = (df["Inventory Level"] == 0).astype(int)
    df["reorder_event"]    = (df["Units Ordered"] > 0).astype(int)
    df["lead_time_demand"] = df["sales_roll_mean_7"].clip(lower=0) * df["Lead Time Days"]

    # ── Promotional & calendar ────────────────────────────────────────────────
    df["discount_x_holiday"] = df["Discount"] * df["Holiday/Promotion"]
    df["dow"]        = df["Date"].dt.dayofweek
    df["month"]      = df["Date"].dt.month
    df["is_weekend"] = (df["Date"].dt.dayofweek >= 5).astype(int)

    # ── Categorical encoding ──────────────────────────────────────────────────
    df["Category_enc"] = pd.Categorical(
        df["Category"],
        categories=["Groceries", "Snacks", "Beverages", "Household", "Personal Care"],
    ).codes
    df["Region_enc"] = (
        df["Region"].map({"North": 0, "South": 1, "East": 2, "West": 3, "Central": 4})
        .fillna(-1).astype(int)
    )
    df["Seasonality_enc"] = (
        df["Seasonality"].map({"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3})
        .fillna(-1).astype(int)
    )

    # ── Baseline prediction & sample weight ──────────────────────────────────
    df["y_pred_baseline"] = df["sales_lag_1"].fillna(df["sales_lag_7"])
    store_freq = df["Store ID"].value_counts(normalize=True)
    df["sample_weight"] = (
        df["Store ID"].map(lambda x: 1.0 / max(store_freq.get(x, 0), 1e-6))
        .clip(0.1, 10.0)
    )

    # ── Label & metadata ──────────────────────────────────────────────────────
    df["y"]          = gs.shift(-HORIZON)
    df["as_of_date"] = df["Date"]
    df["series_id"]  = df["Store ID"].astype(str) + "_" + df["Product ID"].astype(str)

    # ── Drop warm-up rows (lag_28) and rows missing a label ──────────────────
    rows_before = len(df)
    df = df.dropna(subset=["sales_lag_28", "y"]).reset_index(drop=True)
    log.info(
        "Dropped %d warm-up / label rows → %d rows remaining",
        rows_before - len(df), len(df),
    )

    log.info(
        "Feature engineering complete — %d rows × %d cols", len(df), len(df.columns)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Chronological 80 / 10 / 10 split
# ─────────────────────────────────────────────────────────────────────────────
def split_data(
    df:         pd.DataFrame,
    train_frac: float = 0.80,
    val_frac:   float = 0.10,
    date_col:   str   = "as_of_date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    dates         = df[date_col].sort_values().unique()
    n             = len(dates)
    train_end     = dates[int(n * train_frac) - 1]
    val_end       = dates[int(n * (train_frac + val_frac)) - 1]

    train = df[df[date_col] <= train_end].copy()
    val   = df[(df[date_col] > train_end) & (df[date_col] <= val_end)].copy()
    test  = df[df[date_col] > val_end].copy()

    log.info(
        "Split — train: %d rows (ends %s) | val: %d rows (ends %s) | test: %d rows (ends %s)",
        len(train), train_end.date(),
        len(val),   val_end.date(),
        len(test),  df[date_col].max().date(),
    )
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=== Local Data Preparation ===")

    # ── Check CSVs exist ──────────────────────────────────────────────────────
    for p in [RETAIL_CSV, SNAPSHOT_CSV]:
        if not p.exists():
            raise FileNotFoundError(
                f"CSV not found: {p}\n"
                f"Make sure both CSVs are in the same folder as this script:\n"
                f"  {RETAIL_CSV.name}\n"
                f"  {SNAPSHOT_CSV.name}"
            )

    # ── Step 1: build features ────────────────────────────────────────────────
    df = build_features(RETAIL_CSV, SNAPSHOT_CSV)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    features_path = DATA_DIR / "features.parquet"
    df.to_parquet(features_path, index=False)
    log.info("Saved → %s", features_path)

    # ── Step 2: split ─────────────────────────────────────────────────────────
    train, val, test = split_data(df)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(SPLITS_DIR / "train.parquet", index=False)
    val.to_parquet(SPLITS_DIR   / "val.parquet",   index=False)
    test.to_parquet(SPLITS_DIR  / "test.parquet",  index=False)

    log.info("Saved → %s", SPLITS_DIR / "train.parquet")
    log.info("Saved → %s", SPLITS_DIR / "val.parquet")
    log.info("Saved → %s", SPLITS_DIR / "test.parquet")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=== All done — folder structure ===")
    log.info("data/")
    log.info("  features.parquet       (%d rows)", len(df))
    log.info("  splits/")
    log.info("    train.parquet        (%d rows)", len(train))
    log.info("    val.parquet          (%d rows)", len(val))
    log.info("    test.parquet         (%d rows)", len(test))
    log.info("")
    log.info("You can now run:")
    log.info("  python prophet_model.py")
    log.info("  python xgboost_model.py")
    log.info("  python lstm_model.py")


if __name__ == "__main__":
    main()