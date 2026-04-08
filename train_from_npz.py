"""
train_from_npz.py
=================
Load pre-generated synthetic_dataset.npz and train + save all models as .pkl

Usage:
    python train_from_npz.py
    python train_from_npz.py --npz data/synthetic_dataset.npz --out saved_models
"""

import argparse
import numpy as np
import joblib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not found, using GradientBoosting instead.")


def load_npz(npz_path: str):
    print(f"\n  Loading dataset: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    X           = data["X"]
    y_pm25      = data["y_pm25"]
    y_pm10      = data["y_pm10"]
    rh          = data["rh"]
    
    # feature_names might be stored as object array — flatten it
    raw_names = data["feature_names"]
    if raw_names.ndim == 0:
        feature_names = list(raw_names.item())
    else:
        feature_names = list(raw_names)

    print(f"  Samples      : {X.shape[0]}")
    print(f"  Features     : {X.shape[1]}")
    print(f"  PM2.5 range  : {y_pm25.min():.1f} — {y_pm25.max():.1f} µg/m³")
    print(f"  PM10 range   : {y_pm10.min():.1f} — {y_pm10.max():.1f} µg/m³")

    return X, y_pm25, y_pm10, rh, feature_names


def build_models():
    models = {}

    models["RandomForest"] = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )

    if HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    else:
        models["GradientBoosting"] = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            random_state=42,
        )

    models["NeuralNetwork"] = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=200,
        batch_size=64,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )

    return models


def train_and_save(npz_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────
    X, y_pm25, y_pm10, rh, feature_names = load_npz(npz_path)

    # ── 2. Train/test split ──────────────────────────────────────────
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test  = scaler.transform(X[test_idx])

    y_train = {"pm25": y_pm25[train_idx], "pm10": y_pm10[train_idx]}
    y_test  = {"pm25": y_pm25[test_idx],  "pm10": y_pm10[test_idx]}

    print(f"\n  Train samples : {len(train_idx)}")
    print(f"  Test samples  : {len(test_idx)}")

    # ── 3. Save scaler ───────────────────────────────────────────────
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"\n  Saved scaler  : {scaler_path}")

    # ── 4. Train each model for PM2.5 and PM10 ───────────────────────
    model_templates = build_models()
    all_results = {}

    for target in ["pm25", "pm10"]:
        print(f"\n  {'─'*50}")
        print(f"  Training for {target.upper()}")
        print(f"  {'─'*50}")

        for name, template in model_templates.items():
            from sklearn.base import clone
            model = clone(template)

            model.fit(X_train, y_train[target])
            y_pred = model.predict(X_test)

            mae  = mean_absolute_error(y_test[target], y_pred)
            rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
            r2   = r2_score(y_test[target], y_pred)

            print(f"    {name:>20s}  MAE={mae:7.2f}  RMSE={rmse:7.2f}  R²={r2:.4f}")

            # Save model
            model_path = os.path.join(output_dir, f"{name}_{target}.pkl")
            joblib.dump(model, model_path)
            print(f"    {'':>20s}  Saved → {model_path}")

            if name not in all_results:
                all_results[name] = {}
            all_results[name][target] = {
                "mae": round(float(mae), 4),
                "rmse": round(float(rmse), 4),
                "r2": round(float(r2), 4),
            }

    # ── 5. Save metadata ─────────────────────────────────────────────
    meta = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "models_available": [
            f"{name}_{target}"
            for name in model_templates
            for target in ["pm25", "pm10"]
        ],
        "results": all_results,
    }

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved metadata : {meta_path}")

    # ── 6. Summary ───────────────────────────────────────────────────
    print(f"\n  {'═'*50}")
    print(f"  DONE — models saved to: {output_dir}/")
    print(f"  {'═'*50}")
    print(f"  {'Model':>20s}  {'PM2.5 MAE':>12s}  {'PM2.5 R²':>10s}")
    print(f"  {'─'*50}")
    for name, metrics in all_results.items():
        print(f"  {name:>20s}  {metrics['pm25']['mae']:>12.2f}  {metrics['pm25']['r2']:>10.4f}")

    print(f"\n  Files saved:")
    for f in sorted(os.listdir(output_dir)):
        size_kb = os.path.getsize(os.path.join(output_dir, f)) / 1024
        print(f"    {f:<35s}  {size_kb:8.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/synthetic_dataset.npz",
                        help="Path to .npz dataset file")
    parser.add_argument("--out", default="saved_models",
                        help="Output directory for .pkl files")
    args = parser.parse_args()

    train_and_save(args.npz, args.out)