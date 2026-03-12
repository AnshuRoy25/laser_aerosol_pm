"""
Feature Engineering
===================
Extracts physics-informed features from the 9-channel raw scattering
signals plus environmental data. Produces the 25–30 dimensional
feature vector used by ML models.

Feature categories:
    1. Raw channels (9): 3λ × 3θ scattering voltages
    2. Spectral ratios (9): wavelength pair ratios at each angle
    3. Angular ratios (6): angle pair ratios at each wavelength
    4. Ångström-like exponents (3): from spectral ratios
    5. Aggregate features (3): total, fine-fraction, coarse-fraction
    6. Environmental (4): T, RH, dew point, absolute humidity
"""

import numpy as np


class FeatureEngineer:
    """
    Transform raw 3×3 scattering matrix + environment into ML features.
    
    Usage:
        fe = FeatureEngineer()
        features, names = fe.extract(voltages_3x3, temperature, rh)
    """
    
    def __init__(self):
        self.feature_names = self._build_feature_names()
    
    def extract(
        self,
        voltages: np.ndarray,
        temperature: float,
        rh: float
    ) -> tuple:
        """
        Extract complete feature vector from sensor reading.
        
        Parameters
        ----------
        voltages : np.ndarray, shape (3, 3)
            Scattering voltages [wavelength_idx, angle_idx].
            Wavelengths: [450, 532, 650] nm
            Angles: [30°, 60°, 90°]
        temperature : float
            Temperature in °C.
        rh : float
            Relative humidity in %.
            
        Returns
        -------
        (features, names) : tuple
            features: 1D array of extracted features.
            names: list of feature names.
        """
        features = []
        
        # Avoid division by zero
        v = np.clip(voltages, 1e-12, None)
        
        # ── 1. Raw channels (9 features) ────────────────────────────────
        for i in range(3):
            for j in range(3):
                features.append(v[i, j])
        
        # ── 2. Spectral ratios (9 features) ─────────────────────────────
        # S(450)/S(650), S(532)/S(650), S(450)/S(532) at each angle
        for j in range(3):
            features.append(v[0, j] / v[2, j])   # 450/650
            features.append(v[1, j] / v[2, j])   # 532/650
            features.append(v[0, j] / v[1, j])   # 450/532
        
        # ── 3. Angular ratios (6 features) ──────────────────────────────
        # S(30°)/S(90°), S(60°)/S(90°) at each wavelength
        for i in range(3):
            features.append(v[i, 0] / v[i, 2])   # 30°/90°
            features.append(v[i, 1] / v[i, 2])   # 60°/90°
        
        # ── 4. Ångström-like exponents (3 features) ─────────────────────
        # α = -ln(S₁/S₂) / ln(λ₁/λ₂) at each angle
        wl = np.array([450, 532, 650], dtype=float)
        for j in range(3):
            if v[0, j] > 0 and v[2, j] > 0:
                alpha = -np.log(v[0, j] / v[2, j]) / np.log(wl[0] / wl[2])
            else:
                alpha = 0.0
            features.append(alpha)
        
        # ── 5. Aggregate features (3 features) ──────────────────────────
        # Total scattering
        total = np.sum(v)
        features.append(total)
        
        # Fine-fraction proxy: ratio of 450nm to 650nm (all angles summed)
        fine_frac = np.sum(v[0, :]) / max(np.sum(v[2, :]), 1e-12)
        features.append(fine_frac)
        
        # Forward-to-side ratio (all wavelengths) — particle size indicator
        fwd_side = np.sum(v[:, 0]) / max(np.sum(v[:, 2]), 1e-12)
        features.append(fwd_side)
        
        # ── 6. Environmental features (4 features) ──────────────────────
        features.append(temperature)
        features.append(rh)
        
        # Dew point (Magnus formula)
        a, b = 17.27, 237.7
        gamma = (a * temperature / (b + temperature)) + np.log(max(rh, 1) / 100.0)
        dew_point = (b * gamma) / (a - gamma)
        features.append(dew_point)
        
        # Absolute humidity (g/m³)
        abs_humidity = (6.112 * np.exp(a * temperature / (b + temperature)) * rh * 2.1674) / (273.15 + temperature)
        features.append(abs_humidity)
        
        return np.array(features), self.feature_names
    
    def extract_batch(
        self,
        voltages_list: list,
        temperatures: np.ndarray,
        rh_values: np.ndarray
    ) -> tuple:
        """
        Extract features for multiple samples.
        
        Returns
        -------
        (X, names) : tuple
            X: shape (n_samples, n_features)
            names: list of feature names
        """
        all_features = []
        for v, t, r in zip(voltages_list, temperatures, rh_values):
            feat, names = self.extract(v, t, r)
            all_features.append(feat)
        
        return np.array(all_features), names
    
    def _build_feature_names(self) -> list:
        """Build ordered list of feature names."""
        names = []
        wl_labels = ["450nm", "532nm", "650nm"]
        ang_labels = ["30deg", "60deg", "90deg"]
        
        # Raw channels
        for wl in wl_labels:
            for ang in ang_labels:
                names.append(f"V_{wl}_{ang}")
        
        # Spectral ratios
        for ang in ang_labels:
            names.append(f"SR_450_650_{ang}")
            names.append(f"SR_532_650_{ang}")
            names.append(f"SR_450_532_{ang}")
        
        # Angular ratios
        for wl in wl_labels:
            names.append(f"AR_30_90_{wl}")
            names.append(f"AR_60_90_{wl}")
        
        # Ångström exponents
        for ang in ang_labels:
            names.append(f"AE_450_650_{ang}")
        
        # Aggregates
        names.extend(["total_scattering", "fine_fraction_proxy", "fwd_side_ratio"])
        
        # Environmental
        names.extend(["temperature_C", "rh_pct", "dew_point_C", "abs_humidity_gm3"])
        
        return names
    
    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def demo():
    """Demonstrate feature engineering."""
    fe = FeatureEngineer()
    
    print("=" * 70)
    print("FEATURE ENGINEERING — DEMO")
    print("=" * 70)
    
    # Synthetic voltage reading
    voltages = np.array([
        [0.85, 0.42, 0.18],
        [0.72, 0.35, 0.14],
        [0.55, 0.28, 0.10],
    ])
    
    features, names = fe.extract(voltages, temperature=25.0, rh=65.0)
    
    print(f"\n  Total features: {fe.n_features}")
    print(f"\n  Feature vector:")
    for name, val in zip(names, features):
        print(f"    {name:>25s} = {val:12.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
