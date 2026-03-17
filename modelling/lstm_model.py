"""
lstm_model.py  —  pipeline version 2.0  (PyTorch backend)
Demand forecasting using a multi-series LSTM.

Rewritten from TensorFlow to PyTorch for Windows CPU compatibility.
Architecture and behaviour are identical to the TF version.

Architecture
------------
  Input  : (batch, SEQ_LEN, n_features)
  Layer 1: LSTM(128 units, return_sequences=True)  + Dropout(0.2)
  Layer 2: LSTM(64  units, return_sequences=False) + Dropout(0.2)
  Layer 3: Linear(64 → 32) + ReLU
  Output : Linear(32 → 1)

Usage
-----
    python lstm_model.py
    python lstm_model.py --epochs 50 --seq-len 28
"""

from __future__ import annotations

# ── Windows DLL fix — must be before ANY other import ────────────────────────
# PyTorch on Windows needs System32 on the DLL search path.
# This is a no-op on Linux/Mac.
import os
import sys
if sys.platform == "win32":
    os.add_dll_directory(r"C:\Windows\System32")

import argparse
import json
import logging
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise ImportError(
        "Install PyTorch:  pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from e

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: %s", DEVICE)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28",
    "sales_roll_std_7", "sales_ewm_28",
    "demand_forecast_lag1",
    "price_vs_competitor", "effective_price",
    "Holiday/Promotion", "Discount", "discount_x_holiday",
    "dow", "month", "is_weekend",
    "Inventory Level", "stockout_flag", "lead_time_demand",
    "Lead Time Days", "reorder_event",
    "Category_enc", "Region_enc", "Seasonality_enc",
    "series_enc",
]

NO_SCALE_COLS = [
    "Holiday/Promotion", "Discount", "discount_x_holiday",
    "dow", "month", "is_weekend", "stockout_flag", "reorder_event",
    "Category_enc", "Region_enc", "Seasonality_enc",
    "Lead Time Days", "series_enc",
]

TARGET_COL = "y"
DATE_COL   = "as_of_date"
WEIGHT_COL = "sample_weight"

SEQ_LEN    = 28
BATCH_SIZE = 512
EPOCHS     = 50
PATIENCE   = 10
LR         = 1e-3

DEFAULT_DATA_DIR    = Path("data/splits")
DEFAULT_OUTPUT_DIR  = Path("data/models/lstm")
DEFAULT_REPORTS_DIR = Path("reports")


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = y_true > 0
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2   = r2_score(y_true, y_pred)

    metrics = {
        "mae":  round(float(mae),  4),
        "rmse": round(float(rmse), 4),
        "mape": round(float(mape), 4),
        "r2":   round(float(r2),   4),
    }
    if label:
        log.info(
            "%s → MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  R²=%.4f",
            label, mae, rmse, mape, r2,
        )
    return metrics


# ── Series encoding ───────────────────────────────────────────────────────────
def encode_series(train: pd.DataFrame, *others: pd.DataFrame) -> tuple:
    if "series_id" not in train.columns:
        train = train.copy()
        train["series_id"] = (
            train["Store ID"].astype(str) + "_" + train["Product ID"].astype(str)
        )
    mapping = {sid: i for i, sid in enumerate(sorted(train["series_id"].unique()))}
    results = []
    for df in (train, *others):
        df = df.copy()
        if "series_id" not in df.columns:
            df["series_id"] = (
                df["Store ID"].astype(str) + "_" + df["Product ID"].astype(str)
            )
        df["series_enc"] = df["series_id"].map(mapping).fillna(-1).astype(int)
        results.append(df)
    return tuple(results)


# ── Scaling ───────────────────────────────────────────────────────────────────
def fit_scaler(
    train: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    scale_cols = [
        c for c in feature_cols
        if c in train.columns and c not in NO_SCALE_COLS
    ]
    scaler = StandardScaler()
    train  = train.copy()
    train[scale_cols] = scaler.fit_transform(train[scale_cols])
    log.info("Scaler fit on %d columns", len(scale_cols))
    return train, scaler, scale_cols


def apply_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    scale_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in scale_cols if c in df.columns]
    df[cols] = scaler.transform(df[cols])
    return df


# ── Sequence builder ──────────────────────────────────────────────────────────
def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window sequences built per series.
    Returns X (n_samples, seq_len, n_features) and y (n_samples,).
    """
    available = [c for c in feature_cols if c in df.columns]
    all_X, all_y = [], []

    if "series_id" not in df.columns:
        df = df.copy()
        df["series_id"] = (
            df["Store ID"].astype(str) + "_" + df["Product ID"].astype(str)
        )

    for sid in df["series_id"].unique():
        s = df[df["series_id"] == sid].sort_values(DATE_COL).reset_index(drop=True)
        feat = s[available].values.astype(np.float32)
        tgt  = s[TARGET_COL].values.astype(np.float32)

        if len(s) <= seq_len:
            continue

        for i in range(seq_len, len(s)):
            all_X.append(feat[i - seq_len : i])
            all_y.append(tgt[i])

    if not all_X:
        raise ValueError("No sequences built — check seq_len vs series length.")

    X = np.stack(all_X, axis=0)
    y = np.array(all_y, dtype=np.float32)
    log.info("Built %d sequences  shape=%s", len(y), X.shape)
    return X, y


# ── PyTorch model ─────────────────────────────────────────────────────────────
class LSTMForecast(nn.Module):
    """
    Two-layer stacked LSTM with a small dense head.
    Mirrors the TensorFlow architecture exactly.
    """
    def __init__(self, n_features: int, hidden1: int = 128, hidden2: int = 64,
                 dense: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm1   = nn.LSTM(n_features, hidden1, batch_first=True)
        self.drop1   = nn.Dropout(dropout)
        self.lstm2   = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2   = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden2, dense)
        self.relu    = nn.ReLU()
        self.fc_out  = nn.Linear(dense, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)           # (batch, seq, hidden1)
        out    = self.drop1(out)
        out, _ = self.lstm2(out)         # (batch, seq, hidden2)
        out    = self.drop2(out)
        out    = out[:, -1, :]           # take last timestep → (batch, hidden2)
        out    = self.relu(self.fc1(out))
        out    = self.fc_out(out)        # (batch, 1)
        return out.squeeze(1)            # (batch,)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    output_dir: Path,
    epochs:     int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tuple[LSTMForecast, dict]:

    n_features = X_train.shape[2]
    model = LSTMForecast(n_features=n_features).to(DEVICE)
    log.info(
        "Model params: %d",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # DataLoaders
    def make_loader(X, y, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)

    criterion = nn.L1Loss()          # MAE loss — robust to retail outliers
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0
    history        = {"train_loss": [], "val_loss": []}

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "lstm_best.pt"

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                val_losses.append(criterion(preds, y_batch).item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))

        scheduler.step(val_loss)

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f",
            epoch, epochs, train_loss, val_loss,
        )

        # ── Early stopping ─────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            torch.save(best_state, checkpoint_path)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                log.info("Early stopping at epoch %d (patience=%d)", epoch, PATIENCE)
                break

    # Restore best weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    log.info("Best epoch: %d  best_val_loss=%.4f", best_epoch, best_val_loss)

    history["best_epoch"]      = best_epoch
    history["best_val_loss"]   = round(best_val_loss, 4)
    return model, history


# ── Inference helper ──────────────────────────────────────────────────────────
def predict(model: LSTMForecast, X: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    preds = []
    with torch.no_grad():
        for (X_batch,) in loader:
            preds.append(model(X_batch.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds).clip(min=0)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(
    data_dir:    Path,
    output_dir:  Path,
    reports_dir: Path,
    seq_len:     int = SEQ_LEN,
    epochs:      int = EPOCHS,
) -> None:
    log.info("=== LSTM Model Training  v2.0  (PyTorch) ===")
    log.info("SEQ_LEN=%d  EPOCHS=%d  BATCH=%d  DEVICE=%s",
             seq_len, epochs, BATCH_SIZE, DEVICE)

    # ── Load splits ───────────────────────────────────────────────────────────
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df   = pd.read_parquet(data_dir / "val.parquet")
    test_df  = pd.read_parquet(data_dir / "test.parquet")
    log.info(
        "Loaded  train=%d  val=%d  test=%d rows",
        len(train_df), len(val_df), len(test_df),
    )

    # ── Encode series ─────────────────────────────────────────────────────────
    train_df, val_df, test_df = encode_series(train_df, val_df, test_df)

    # ── Scale (fit on train only) ─────────────────────────────────────────────
    train_df, scaler, scale_cols = fit_scaler(train_df, FEATURE_COLS)
    val_df  = apply_scaler(val_df,  scaler, scale_cols)
    test_df = apply_scaler(test_df, scaler, scale_cols)

    # ── Build sequences ───────────────────────────────────────────────────────
    log.info("Building sequences (seq_len=%d) ...", seq_len)
    X_train, y_train = build_sequences(train_df, FEATURE_COLS, seq_len)
    X_val,   y_val   = build_sequences(val_df,   FEATURE_COLS, seq_len)
    X_test,  y_test  = build_sequences(test_df,  FEATURE_COLS, seq_len)
    log.info(
        "Sequences — train: %s  val: %s  test: %s",
        X_train.shape, X_val.shape, X_test.shape,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        output_dir, epochs, BATCH_SIZE,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_pred  = predict(model, X_val)
    test_pred = predict(model, X_test)

    val_metrics  = compute_metrics(y_val,  val_pred,  label="LSTM val")
    test_metrics = compute_metrics(y_test, test_pred, label="LSTM test")

    # ── Save model & scaler ───────────────────────────────────────────────────
    model_path = output_dir / "lstm_final.pt"
    torch.save(model.state_dict(), model_path)
    log.info("Model saved to %s", model_path)

    scaler_path = output_dir / "lstm_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "scale_cols": scale_cols}, f)
    log.info("Scaler saved to %s", scaler_path)

    # ── Save training curve ───────────────────────────────────────────────────
    curve_path = output_dir / "lstm_training_curve.json"
    with open(curve_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Save report ───────────────────────────────────────────────────────────
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model":            "lstm",
        "backend":          "pytorch",
        "pipeline_version": "2.0",
        "architecture": {
            "seq_len":    seq_len,
            "n_features": X_train.shape[2],
            "lstm_units": [128, 64],
            "dropout":    0.2,
            "dense":      32,
        },
        "training": {
            "epochs_run":       len(history["train_loss"]),
            "best_epoch":       history["best_epoch"],
            "batch_size":       BATCH_SIZE,
            "learning_rate":    LR,
            "best_val_loss":    history["best_val_loss"],
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss":   history["val_loss"][-1],
        },
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
        "timestamp":    datetime.utcnow().isoformat() + "Z",
    }
    report_path = reports_dir / "lstm_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report saved to %s", report_path)

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=== LSTM Results ===")
    log.info("Val  — MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  R²=%.4f",
             val_metrics["mae"], val_metrics["rmse"],
             val_metrics["mape"], val_metrics["r2"])
    log.info("Test — MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  R²=%.4f",
             test_metrics["mae"], test_metrics["rmse"],
             test_metrics["mape"], test_metrics["r2"])
    log.info("=== Done ===")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM demand forecasting model (PyTorch)")
    parser.add_argument("--data-dir",    type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--seq-len",     type=int,  default=SEQ_LEN)
    parser.add_argument("--epochs",      type=int,  default=EPOCHS)
    args = parser.parse_args()

    main(args.data_dir, args.output_dir, args.reports_dir, args.seq_len, args.epochs)