"""
ML Training Pipeline
====================
Trains and evaluates PM2.5/PM10 prediction models:
    - Random Forest (primary, deployable on ESP32)
    - XGBoost (highest accuracy)
    - Neural Network (sklearn MLPRegressor, TFLite-compatible)

Includes:
    - Train/test split with stratification by scenario
    - 5-fold cross-validation
    - Feature importance analysis
    - Ablation study (removing feature groups)
    - Humidity-binned performance analysis
"""

import numpy as np
import json
import os
from time import time

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed, using sklearn GradientBoosting instead.")

from config import (
    TEST_SIZE, RANDOM_STATE,
    RF_N_ESTIMATORS, RF_MAX_DEPTH,
    XGB_N_ESTIMATORS, XGB_LEARNING_RATE, XGB_MAX_DEPTH,
    NN_HIDDEN_LAYERS, NN_EPOCHS, NN_BATCH_SIZE
)


class MiniLidarMLPipeline:
    """
    Complete ML pipeline for PM2.5/PM10 prediction.
    
    Usage:
        pipeline = MiniLidarMLPipeline()
        pipeline.load_data(X, y_pm25, y_pm10, rh_values, feature_names)
        results = pipeline.train_all()
        pipeline.ablation_study()
        pipeline.humidity_analysis()
    """
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.X_train = None
        self.X_test = None
        self.y_train = {}
        self.y_test = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_names = None
        self.rh_test = None
        
    def load_data(
        self,
        X: np.ndarray,
        y_pm25: np.ndarray,
        y_pm10: np.ndarray,
        rh_values: np.ndarray = None,
        feature_names: list = None
    ):
        """Load and split data."""
        self.feature_names = feature_names
        
        # Split
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(X[train_idx])
        self.X_test = self.scaler.transform(X[test_idx])
        
        self.y_train = {"pm25": y_pm25[train_idx], "pm10": y_pm10[train_idx]}
        self.y_test = {"pm25": y_pm25[test_idx], "pm10": y_pm10[test_idx]}
        
        if rh_values is not None:
            self.rh_test = rh_values[test_idx]
        
        print(f"  Data loaded: {len(train_idx)} train, {len(test_idx)} test")
        print(f"  Features: {X.shape[1]}")
        print(f"  PM2.5 range: {y_pm25.min():.1f} — {y_pm25.max():.1f} µg/m³")
        print(f"  PM10 range:  {y_pm10.min():.1f} — {y_pm10.max():.1f} µg/m³")
    
    def _build_models(self) -> dict:
        """Create model instances."""
        models = {}
        
        # Random Forest
        models["RandomForest"] = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        
        # XGBoost or fallback
        if HAS_XGB:
            models["XGBoost"] = xgb.XGBRegressor(
                n_estimators=XGB_N_ESTIMATORS,
                learning_rate=XGB_LEARNING_RATE,
                max_depth=XGB_MAX_DEPTH,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbosity=0,
            )
        else:
            models["GradientBoosting"] = GradientBoostingRegressor(
                n_estimators=XGB_N_ESTIMATORS,
                learning_rate=XGB_LEARNING_RATE,
                max_depth=XGB_MAX_DEPTH,
                subsample=0.8,
                random_state=RANDOM_STATE,
            )
        
        # Neural Network
        models["NeuralNetwork"] = MLPRegressor(
            hidden_layer_sizes=tuple(NN_HIDDEN_LAYERS),
            activation="relu",
            max_iter=NN_EPOCHS,
            batch_size=NN_BATCH_SIZE,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        )
        
        return models
    
    def train_all(self) -> dict:
        """
        Train all models for both PM2.5 and PM10.
        
        Returns
        -------
        results : dict
            Nested dict: results[model_name][target] = {mae, rmse, r2, time}
        """
        model_templates = self._build_models()
        results = {}
        
        for target in ["pm25", "pm10"]:
            print(f"\n  {'─' * 50}")
            print(f"  Training models for {target.upper()}")
            print(f"  {'─' * 50}")
            
            for name, template in model_templates.items():
                # Clone model for each target
                from sklearn.base import clone
                model = clone(template)
                
                t0 = time()
                model.fit(self.X_train, self.y_train[target])
                train_time = time() - t0
                
                # Predict
                y_pred = model.predict(self.X_test)
                
                # Metrics
                mae = mean_absolute_error(self.y_test[target], y_pred)
                rmse = np.sqrt(mean_squared_error(self.y_test[target], y_pred))
                r2 = r2_score(self.y_test[target], y_pred)
                
                # Store
                model_key = f"{name}_{target}"
                self.models[model_key] = model
                
                if name not in results:
                    results[name] = {}
                results[name][target] = {
                    "mae": mae, "rmse": rmse, "r2": r2, "time_s": train_time
                }
                
                print(f"    {name:>20s}:  MAE={mae:7.2f}  RMSE={rmse:7.2f}  R²={r2:.4f}  ({train_time:.1f}s)")
        
        self.results = results
        return results
    
    def cross_validate(self, n_folds: int = 5) -> dict:
        """Run k-fold cross-validation for all models."""
        model_templates = self._build_models()
        cv_results = {}
        
        print(f"\n  {'─' * 50}")
        print(f"  {n_folds}-Fold Cross-Validation (PM2.5)")
        print(f"  {'─' * 50}")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        
        for name, model in model_templates.items():
            scores = cross_val_score(
                model, self.X_train, self.y_train["pm25"],
                cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1
            )
            mae_scores = -scores
            cv_results[name] = {
                "mae_mean": mae_scores.mean(),
                "mae_std":  mae_scores.std(),
                "mae_folds": mae_scores.tolist(),
            }
            print(f"    {name:>20s}:  MAE = {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")
        
        return cv_results
    
    def feature_importance(self, target: str = "pm25") -> dict:
        """
        Extract feature importance from Random Forest model.
        
        Returns sorted dict: {feature_name: importance}.
        """
        model_key = f"RandomForest_{target}"
        if model_key not in self.models:
            print("  [ERROR] Train RandomForest first.")
            return {}
        
        model = self.models[model_key]
        importances = model.feature_importances_
        
        if self.feature_names is not None:
            names = self.feature_names
        else:
            names = [f"feature_{i}" for i in range(len(importances))]
        
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]
        result = {names[i]: importances[i] for i in sorted_idx}
        
        print(f"\n  {'─' * 50}")
        print(f"  Feature Importance ({target.upper()})")
        print(f"  {'─' * 50}")
        for i, (name, imp) in enumerate(result.items()):
            bar = "█" * int(imp * 100)
            print(f"    {i+1:2d}. {name:>25s}  {imp:.4f}  {bar}")
            if i >= 14:
                print(f"    ... ({len(result) - 15} more features)")
                break
        
        return result
    
    def ablation_study(self, target: str = "pm25") -> dict:
        """
        Ablation study: remove feature groups to quantify contributions.
        
        Groups:
            - raw: 9 raw channels
            - spectral: 9 spectral ratios
            - angular: 6 angular ratios
            - angstrom: 3 Ångström exponents
            - aggregate: 3 aggregate features
            - environmental: 4 environmental features
        """
        feature_groups = {
            "raw":           list(range(0, 9)),
            "spectral":      list(range(9, 18)),
            "angular":       list(range(18, 24)),
            "angstrom":      list(range(24, 27)),
            "aggregate":     list(range(27, 30)),
            "environmental": list(range(30, 34)),
        }
        
        n_feat = self.X_train.shape[1]
        all_indices = list(range(n_feat))
        
        print(f"\n  {'─' * 60}")
        print(f"  ABLATION STUDY — {target.upper()}")
        print(f"  {'─' * 60}")
        
        # Full model baseline
        rf = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
        )
        rf.fit(self.X_train, self.y_train[target])
        y_pred_full = rf.predict(self.X_test)
        mae_full = mean_absolute_error(self.y_test[target], y_pred_full)
        r2_full = r2_score(self.y_test[target], y_pred_full)
        
        print(f"    {'Full model':>25s}:  MAE = {mae_full:.2f}  R² = {r2_full:.4f}")
        
        ablation_results = {"full": {"mae": mae_full, "r2": r2_full}}
        
        for group_name, group_indices in feature_groups.items():
            # Remove this group
            remaining = [i for i in all_indices if i not in group_indices]
            
            if len(remaining) == 0:
                continue
            
            rf_ablated = RandomForestRegressor(
                n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
            )
            rf_ablated.fit(self.X_train[:, remaining], self.y_train[target])
            y_pred_abl = rf_ablated.predict(self.X_test[:, remaining])
            mae_abl = mean_absolute_error(self.y_test[target], y_pred_abl)
            r2_abl = r2_score(self.y_test[target], y_pred_abl)
            
            delta_mae = mae_abl - mae_full
            delta_pct = (delta_mae / mae_full) * 100
            
            ablation_results[f"remove_{group_name}"] = {
                "mae": mae_abl, "r2": r2_abl,
                "delta_mae": delta_mae, "delta_pct": delta_pct,
            }
            
            arrow = "↑" if delta_mae > 0 else "↓"
            print(f"    Remove {group_name:>15s}:  MAE = {mae_abl:.2f}  R² = {r2_abl:.4f}  (MAE {arrow} {abs(delta_pct):.1f}%)")
        
        return ablation_results
    
    def humidity_analysis(self, target: str = "pm25") -> dict:
        """
        Analyse model performance across RH bins.
        Quantifies the humidity compensation effectiveness.
        """
        if self.rh_test is None:
            print("  [ERROR] No RH values in test set.")
            return {}
        
        model_key = f"RandomForest_{target}"
        if model_key not in self.models:
            print("  [ERROR] Train models first.")
            return {}
        
        model = self.models[model_key]
        y_pred = model.predict(self.X_test)
        
        # Also compute "baseline" (simple regression on total scattering only)
        # Feature index 27 = total_scattering
        from sklearn.linear_model import LinearRegression
        baseline = LinearRegression()
        baseline.fit(self.X_train[:, 27:28], self.y_train[target])
        y_pred_baseline = baseline.predict(self.X_test[:, 27:28])
        
        rh_bins = [(0, 40), (40, 60), (60, 80), (80, 100)]
        
        print(f"\n  {'─' * 65}")
        print(f"  HUMIDITY ANALYSIS — {target.upper()}")
        print(f"  {'─' * 65}")
        print(f"    {'RH Range':>12s}  {'Baseline MAE':>14s}  {'ML MAE':>10s}  {'Improvement':>12s}")
        
        humidity_results = {}
        
        for rh_lo, rh_hi in rh_bins:
            mask = (self.rh_test >= rh_lo) & (self.rh_test < rh_hi)
            n = mask.sum()
            
            if n < 5:
                continue
            
            mae_ml = mean_absolute_error(self.y_test[target][mask], y_pred[mask])
            mae_baseline = mean_absolute_error(self.y_test[target][mask], y_pred_baseline[mask])
            improvement = mae_baseline / max(mae_ml, 0.01)
            
            label = f"{rh_lo}–{rh_hi}%"
            humidity_results[label] = {
                "n": int(n), "mae_ml": mae_ml, "mae_baseline": mae_baseline,
                "improvement_factor": improvement,
            }
            
            print(f"    {label:>12s}  {mae_baseline:14.2f}  {mae_ml:10.2f}  {improvement:10.1f}×  (n={n})")
        
        return humidity_results
    
    def save_results(self, filename: str = "training_results.json"):
        """Save all results to JSON."""
        # Convert numpy types for JSON serialisation
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        output = {
            "models": {},
            "feature_names": self.feature_names,
        }
        
        for name, metrics in self.results.items():
            output["models"][name] = {}
            for target, vals in metrics.items():
                output["models"][name][target] = {
                    k: convert(v) for k, v in vals.items()
                }
        
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {path}")
