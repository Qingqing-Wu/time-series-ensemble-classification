
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


class Config:
    date_col: str = "date"
    value_col: str = "value"

    horizon: int = 1
    test_size: float = 0.2
    n_splits: int = 5

    # Feature engineering
    lags: Tuple[int, ...] = (1, 2, 3, 5, 10)
    rolling_windows: Tuple[int, ...] = (3, 5, 10, 20)

    # Modeling
    use_xgb: bool = True
    random_state: int = 42

    # Outputs
    out_dir: str = "reports"



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")



# Data Loading
def load_timeseries_csv(
    csv_path: str,
    cfg: Config,
) -> pd.DataFrame:

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if cfg.date_col not in df.columns:
        raise ValueError(f"Missing date column '{cfg.date_col}'. Columns: {list(df.columns)}")
    if cfg.value_col not in df.columns:
        raise ValueError(f"Missing value column '{cfg.value_col}'. Columns: {list(df.columns)}")

    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col]).copy()
    df = df.sort_values(cfg.date_col).reset_index(drop=True)

    df[cfg.value_col] = pd.to_numeric(df[cfg.value_col], errors="coerce")
    df = df.dropna(subset=[cfg.value_col]).copy()

    df = df.set_index(cfg.date_col)
    return df



# Feature Engineering
def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:

    s = df[cfg.value_col].astype(float).copy()
    out = pd.DataFrame(index=df.index)

    # Basic differences
    out["delta_1"] = s.diff(1)
    out["pct_1"] = s.pct_change(1).replace([np.inf, -np.inf], np.nan)

    # Lags of raw value, delta, pct
    for k in cfg.lags:
        out[f"lag_value_{k}"] = s.shift(k)
        out[f"lag_delta_{k}"] = s.diff(1).shift(k)
        out[f"lag_pct_{k}"] = s.pct_change(1).shift(k).replace([np.inf, -np.inf], np.nan)

    # Rolling statistics
    for w in cfg.rolling_windows:
        roll = s.rolling(window=w, min_periods=max(2, w // 2))
        out[f"roll_mean_{w}"] = roll.mean()
        out[f"roll_std_{w}"] = roll.std()
        out[f"roll_min_{w}"] = roll.min()
        out[f"roll_max_{w}"] = roll.max()
        # z-score (value vs rolling mean/std)
        out[f"roll_z_{w}"] = (s - out[f"roll_mean_{w}"]) / (out[f"roll_std_{w}"] + 1e-9)

    # Trend proxy: difference between short and long rolling mean
    if len(cfg.rolling_windows) >= 2:
        short = min(cfg.rolling_windows)
        long = max(cfg.rolling_windows)
        out[f"mean_gap_{short}_{long}"] = out[f"roll_mean_{short}"] - out[f"roll_mean_{long}"]

    # Cleanup
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_target(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    Binary target: whether the series increases over the next `horizon` steps.
    y(t)=1 if value(t+h) > value(t) else 0
    """
    s = df[cfg.value_col].astype(float)
    future = s.shift(-cfg.horizon)
    y = (future > s).astype(int)
    return y


def assemble_dataset(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    X = build_features(df, cfg)
    y = build_target(df, cfg)

    data = X.copy()
    data["target"] = y

    # Drop rows with NaNs from feature creation and horizon shift
    data = data.dropna().copy()
    y_clean = data.pop("target").astype(int)
    X_clean = data
    return X_clean, y_clean

# Models

def make_base_models(cfg: Config) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    models["logit"] = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=cfg.random_state,
        ))
    ])

    models["rf"] = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=cfg.random_state,
            n_jobs=-1,
        ))
    ])

    models["gbdt"] = Pipeline([
        ("clf", GradientBoostingClassifier(
            random_state=cfg.random_state,
        ))
    ])

    if cfg.use_xgb and _HAS_XGB:
        models["xgb"] = Pipeline([
            ("clf", XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=cfg.random_state,
                n_jobs=-1,
                eval_metric="logloss",
            ))
        ])

    return models



# Blending / Stacking (OOF)

def oof_blend_predictions(
    models: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Config
) -> Tuple[pd.DataFrame, pd.Series]:

    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    oof = pd.DataFrame(index=X.index, columns=list(models.keys()), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        for name, pipe in models.items():
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_va)[:, 1]
            oof.loc[X_va.index, name] = proba

    # Drop any rows not filled due to CV mechanics
    mask = oof.notna().all(axis=1)
    return oof.loc[mask].copy(), y.loc[mask].copy()


def fit_blender(oof_X: pd.DataFrame, oof_y: pd.Series, cfg: Config) -> Pipeline:

    blender = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=cfg.random_state,
        ))
    ])
    blender.fit(oof_X, oof_y)
    return blender


def predict_with_blend(
    models: Dict[str, Pipeline],
    blender: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Fit base models on full training and produce blended probabilities on test.
    """
    base_test = pd.DataFrame(index=X_test.index, columns=list(models.keys()), dtype=float)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        base_test[name] = pipe.predict_proba(X_test)[:, 1]

    blend_proba = blender.predict_proba(base_test)[:, 1]
    return blend_proba, base_test



# Evaluation

def evaluate_binary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)

    ba = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = float("nan")

    return {
        "balanced_accuracy": safe_float(ba),
        "roc_auc": safe_float(auc),
        "threshold": safe_float(threshold),
    }


def export_reports(
    out_dir: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    tag: str,
) -> None:
    ensure_dir(out_dir)

    y_pred = (y_proba >= threshold).astype(int)
    metrics = evaluate_binary(y_true, y_proba, threshold=threshold)

    # classification report
    cr = classification_report(y_true, y_pred, digits=4)
    with open(os.path.join(out_dir, f"classification_report_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(cr)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    cm_df.to_csv(os.path.join(out_dir, f"confusion_matrix_{tag}.csv"), index=True)

    # ROC curve points
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})
    roc_df.to_csv(os.path.join(out_dir, f"roc_curve_{tag}.csv"), index=False)

    # metrics json
    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)



# Main

def train_test_split_time_order(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    n = len(X)
    if n < 100:
        raise ValueError(f"Dataset too small after feature engineering: n={n}.")
    cut = int(np.floor(n * (1 - test_size)))
    X_train, X_test = X.iloc[:cut].copy(), X.iloc[cut:].copy()
    y_train, y_test = y.iloc[:cut].copy(), y.iloc[cut:].copy()
    return X_train, X_test, y_train, y_test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="General time-series classification with blending ensemble (portfolio-ready)."
    )
    p.add_argument("--csv", type=str, required=True, help="Path to CSV containing a datetime column and a numeric value column.")
    p.add_argument("--date-col", type=str, default="date", help="Datetime column name.")
    p.add_argument("--value-col", type=str, default="value", help="Numeric series column name.")
    p.add_argument("--horizon", type=int, default=1, help="Prediction horizon in steps.")
    p.add_argument("--test-size", type=float, default=0.2, help="Fraction of data used as test (last part).")
    p.add_argument("--splits", type=int, default=5, help="TimeSeriesSplit folds for OOF blending.")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for binary classification.")
    p.add_argument("--out-dir", type=str, default="reports", help="Output directory for reports.")
    p.add_argument("--no-xgb", action="store_true", help="Disable XGBoost even if installed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        date_col=args.date_col,
        value_col=args.value_col,
        horizon=args.horizon,
        test_size=args.test_size,
        n_splits=args.splits,
        out_dir=args.out_dir,
        use_xgb=(not args.no_xgb),
    )

    # Load
    df = load_timeseries_csv(args.csv, cfg)

    # Build dataset
    X, y = assemble_dataset(df, cfg)

    # Split
    X_train, X_test, y_train, y_test = train_test_split_time_order(X, y, cfg.test_size)

    # Base models
    models = make_base_models(cfg)
    if cfg.use_xgb and not _HAS_XGB:
        print("[Info] XGBoost not installed. Proceeding without XGBClassifier.")

    # OOF predictions for blending
    oof_X, oof_y = oof_blend_predictions(models, X_train, y_train, cfg)

    # Fit blender
    blender = fit_blender(oof_X, oof_y, cfg)

    # Predict blended proba on test
    blend_proba, base_test_proba = predict_with_blend(models, blender, X_train, y_train, X_test)

    # Evaluate + export
    tag = timestamp_tag()
    export_reports(cfg.out_dir, y_test.values, blend_proba, threshold=args.threshold, tag=tag)

    # Console summary
    metrics = evaluate_binary(y_test.values, blend_proba, threshold=args.threshold)
    print("\n=== Test Metrics (Blended) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) and not np.isnan(v) else f"{k}: {v}")

    print(f"\nReports saved to: {os.path.abspath(cfg.out_dir)}")
    print(f"Run tag: {tag}")


if __name__ == "__main__":
    main()
