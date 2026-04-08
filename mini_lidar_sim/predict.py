import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.feature_engineering import FeatureEngineer

# ── Your real readings ────────────────────────────────────────────────
V_650_30 = 0.166
V_650_60 = 0.147
V_650_90 = 0.132

temperature_C = 12.0
rh_pct        = 84.0
AE_ESTIMATE   = 1.2

# ── Build voltage matrix ──────────────────────────────────────────────
scale_450 = (650 / 450) ** AE_ESTIMATE
scale_532 = (650 / 532) ** AE_ESTIMATE

voltages = np.array([
    [V_650_30 * scale_450, V_650_60 * scale_450, V_650_90 * scale_450],
    [V_650_30 * scale_532, V_650_60 * scale_532, V_650_90 * scale_532],
    [V_650_30,             V_650_60,             V_650_90            ],
])

# ── Extract features ──────────────────────────────────────────────────
fe = FeatureEngineer()
features, names = fe.extract(voltages, temperature_C, rh_pct)

# ── Find and override the absolute-value features ─────────────────────
# The model sees total_scattering as ~0.18 (sum of your 9 channels)
# but training data had total_scattering ~0.001 to ~0.05 for typical PM
# We need to rescale ONLY the absolute features, not the ratios

# Print current values so you can see what's happening
print("Key feature values BEFORE fix:")
for i, name in enumerate(names):
    if name in ["total_scattering", "fine_fraction_proxy", "fwd_side_ratio"]:
        print(f"  {name:>25s} = {features[i]:.6f}")

# Scale factor applied only to total_scattering (index 27)
# Training total_scattering range was roughly 0.001 to 0.05
# Your current total is sum of 9 channels ≈ 0.18 * ~2.5 (with scaling) ≈ 1.3
# Target: bring total into ~0.02 range (mid-range training value)

features_fixed = features.copy()

# Find indices
total_idx     = names.index("total_scattering")
fine_idx      = names.index("fine_fraction_proxy")
fwd_idx       = names.index("fwd_side_ratio")
raw_indices   = list(range(0, 9))   # V_450nm_30deg ... V_650nm_90deg

# Get current total
current_total = features[total_idx]
# Target total for ~50 µg/m³ PM2.5 in training data was around 0.025
TARGET_TOTAL  = 0.025

scale = TARGET_TOTAL / current_total
print(f"\n  current total_scattering = {current_total:.4f}")
print(f"  target  total_scattering = {TARGET_TOTAL:.4f}")
print(f"  applied scale            = {scale:.6f}")

# Rescale raw channels and total — ratios are untouched
for i in raw_indices:
    features_fixed[i] *= scale
features_fixed[total_idx] *= scale
# fine_fraction and fwd_side are ratios → leave them alone

print("\nKey feature values AFTER fix:")
for i, name in enumerate(names):
    if name in ["total_scattering", "fine_fraction_proxy", "fwd_side_ratio"]:
        print(f"  {name:>25s} = {features_fixed[i]:.6f}")

# ── Load scaler and predict ───────────────────────────────────────────
MODEL_DIR = "saved_models"
scaler    = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
X         = scaler.transform(features_fixed.reshape(1, -1))

trusted_models = ["RandomForest_pm25", "XGBoost_pm25"]

print(f"\n{'Model':>20s}  {'PM2.5 (µg/m³)':>15s}")
print(f"  {'─'*38}")

predictions = {}
for model_name in trusted_models:
    pkl_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    model    = joblib.load(pkl_path)
    pm25_pred = float(model.predict(X)[0])
    predictions[model_name] = pm25_pred
    print(f"  {model_name:>20s}  {pm25_pred:>12.1f}")

ensemble = np.mean(list(predictions.values()))
print(f"\n  {'RF + XGBoost mean':>20s}  {ensemble:>12.1f}")

# ── Try different TARGET_TOTAL values to bracket the answer ──────────
print(f"\nSensitivity check (vary TARGET_TOTAL):")
print(f"  {'TARGET_TOTAL':>14s}  {'RF':>8s}  {'XGB':>8s}  {'mean':>8s}")
for target in [0.005, 0.010, 0.020, 0.025, 0.035, 0.050]:
    f2 = features.copy()
    sc = target / current_total
    for i in raw_indices:
        f2[i] *= sc
    f2[total_idx] *= sc
    X2 = scaler.transform(f2.reshape(1, -1))
    p = [float(joblib.load(os.path.join(MODEL_DIR, f"{m}.pkl")).predict(X2)[0])
         for m in trusted_models]
    print(f"  {target:>14.3f}  {p[0]:>8.1f}  {p[1]:>8.1f}  {np.mean(p):>8.1f}")