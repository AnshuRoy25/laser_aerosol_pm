"""
Mie Scattering Engine
=====================
Computes angular scattering intensities for spherical particles
at multiple wavelengths using PyMieScatt (primary) and miepython
(cross-validation).

Physics:
    - Solves exact Mie theory for homogeneous spheres
    - Returns S1, S2 scattering matrix elements at specified angles
    - Computes Qsca, Qext, Qabs, asymmetry parameter g
    - Handles core-shell particles for hygroscopic growth modelling
"""

import numpy as np

# Compatibility patch: scipy ≥ 1.14 removed trapz → trapezoid
# PyMieScatt still imports the old name
import scipy.integrate
if not hasattr(scipy.integrate, "trapz"):
    scipy.integrate.trapz = scipy.integrate.trapezoid

import PyMieScatt as pms
import miepython

from config import (
    WAVELENGTHS_UM, WAVELENGTHS_NM,
    DETECTOR_ANGLES_DEG, DETECTOR_ANGLES_RAD,
    REFRACTIVE_INDICES
)

class MieScatteringEngine:
    """
    Multi-wavelength Mie scattering calculator for the Mini-LIDAR sensor.
    
    Computes scattering intensities at the 3 detector angles for each
    of the 3 laser wavelengths, producing a 9-element feature vector
    for a single particle diameter.
    
    Usage:
        engine = MieScatteringEngine()
        signals = engine.compute_scattering(diameter_um=1.0, m=complex(1.53, 0))
        # signals shape: (3 wavelengths, 3 angles)
    """
    
    def __init__(self):
        self.wavelengths_um = np.array(WAVELENGTHS_UM)
        self.wavelengths_nm = np.array(WAVELENGTHS_NM)
        self.angles_deg = np.array(DETECTOR_ANGLES_DEG)
        self.angles_rad = np.array(DETECTOR_ANGLES_RAD)
        
        # Pre-compute angular measurement positions (0-180° full range)
        self.theta_full = np.linspace(0, 180, 1801)  # 0.1° resolution
        
        # Cache for repeated lookups
        self._cache = {}
        
    def compute_scattering(
        self,
        diameter_um: float,
        m: complex = None,
        aerosol_type: str = "ammonium_sulfate"
    ) -> np.ndarray:
        """
        Compute scattering intensity at detector angles for all wavelengths.
        
        Parameters
        ----------
        diameter_um : float
            Particle diameter in micrometres.
        m : complex, optional
            Refractive index. If None, uses aerosol_type lookup.
        aerosol_type : str
            Key into REFRACTIVE_INDICES if m is not provided.
            
        Returns
        -------
        signals : np.ndarray, shape (3, 3)
            Scattering intensity [wavelength_index, angle_index].
            Units: differential scattering cross-section (µm² sr⁻¹).
        """
        if m is None:
            m = REFRACTIVE_INDICES[aerosol_type]
            
        signals = np.zeros((len(self.wavelengths_um), len(self.angles_deg)))
        
        for i, wl_um in enumerate(self.wavelengths_um):
            # Size parameter
            x = np.pi * diameter_um / wl_um
            
            if x < 0.01:
                # Too small → negligible scattering
                continue
                
            for j, angle_deg in enumerate(self.angles_deg):
                signals[i, j] = self._single_angle_intensity(
                    diameter_um, wl_um, m, angle_deg
                )
                
        return signals
    
    def compute_scattering_distribution(
        self,
        diameters_um: np.ndarray,
        number_distribution: np.ndarray,
        m: complex = None,
        aerosol_type: str = "ammonium_sulfate"
    ) -> np.ndarray:
        """
        Integrate scattering over a particle size distribution.
        
        Parameters
        ----------
        diameters_um : array
            Bin centres for diameter.
        number_distribution : array
            dN/dlogD for each bin (number concentration per bin).
        m : complex, optional
            Refractive index.
        aerosol_type : str
            Fallback aerosol type.
            
        Returns
        -------
        total_signals : np.ndarray, shape (3, 3)
            Total scattering intensity integrated over the distribution.
        """
        if m is None:
            m = REFRACTIVE_INDICES[aerosol_type]
            
        total_signals = np.zeros((len(self.wavelengths_um), len(self.angles_deg)))
        
        for k, (d, dN) in enumerate(zip(diameters_um, number_distribution)):
            if dN <= 0:
                continue
            single = self.compute_scattering(d, m=m)
            total_signals += single * dN
            
        return total_signals
    
    def compute_efficiencies(
        self,
        diameter_um: float,
        wavelength_um: float,
        m: complex
    ) -> dict:
        """
        Compute Mie scattering/extinction/absorption efficiencies.
        
        Returns
        -------
        dict with keys: Qext, Qsca, Qabs, Qback, g (asymmetry parameter)
        """
        x = np.pi * diameter_um / wavelength_um
        
        if x < 0.01:
            return {"Qext": 0, "Qsca": 0, "Qabs": 0, "Qback": 0, "g": 0}
            
        # PyMieScatt returns (Qext, Qsca, Qabs, g, Qpr, Qback, Qratio)
        result = pms.MieQ(m, wavelength_um * 1000, diameter_um * 1000)
        
        return {
            "Qext":  result[0],
            "Qsca":  result[1],
            "Qabs":  result[2],
            "g":     result[3],
            "Qback": result[5],
        }
    
    def compute_phase_function(
        self,
        diameter_um: float,
        wavelength_um: float,
        m: complex,
        angles_deg: np.ndarray = None
    ) -> tuple:
        """
        Compute the full scattering phase function P(θ).
        
        Returns
        -------
        (angles_deg, SL, SR, SU) : tuple of arrays
            SL = |S2|² (parallel), SR = |S1|² (perpendicular), SU = (SL+SR)/2
        """
        if angles_deg is None:
            angles_deg = self.theta_full
            
        wl_nm = wavelength_um * 1000
        d_nm = diameter_um * 1000
        
        # PyMieScatt angular scattering
        theta, SL, SR, SU = pms.ScatteringFunction(m, wl_nm, d_nm, minAngle=0, maxAngle=180, angularResolution=0.1)
        
        return theta, SL, SR, SU
    
    def compute_angstrom_exponent(
        self,
        diameter_um: float,
        m: complex,
        wl1_idx: int = 0,
        wl2_idx: int = 2
    ) -> float:
        """
        Compute Ångström exponent from scattering at two wavelengths.
        
        α = -ln(σ₁/σ₂) / ln(λ₁/λ₂)
        
        Fine particles → α > 1.5
        Coarse particles → α < 0.5
        """
        eff1 = self.compute_efficiencies(diameter_um, self.wavelengths_um[wl1_idx], m)
        eff2 = self.compute_efficiencies(diameter_um, self.wavelengths_um[wl2_idx], m)
        
        sigma1 = eff1["Qsca"] * np.pi * (diameter_um / 2) ** 2
        sigma2 = eff2["Qsca"] * np.pi * (diameter_um / 2) ** 2
        
        if sigma1 <= 0 or sigma2 <= 0:
            return 0.0
            
        alpha = -np.log(sigma1 / sigma2) / np.log(
            self.wavelengths_um[wl1_idx] / self.wavelengths_um[wl2_idx]
        )
        
        return alpha
    
    def cross_validate_miepython(
        self,
        diameter_um: float,
        wavelength_um: float,
        m: complex
    ) -> dict:
        """
        Cross-validate PyMieScatt results using miepython.
        
        Returns both libraries' Qext, Qsca, Qabs and relative errors.
        """
        x = np.pi * diameter_um / wavelength_um
        
        # PyMieScatt
        pms_result = pms.MieQ(m, wavelength_um * 1000, diameter_um * 1000)
        
        # miepython (takes m, diameter, wavelength in same units)
        mp_qext, mp_qsca, mp_qback, mp_g = miepython.efficiencies(m, diameter_um, wavelength_um)
        
        return {
            "pymie": {"Qext": pms_result[0], "Qsca": pms_result[1], "Qabs": pms_result[2]},
            "miepy": {"Qext": mp_qext,  "Qsca": mp_qsca,  "Qabs": mp_qext - mp_qsca},
            "relative_error": {
                "Qext": abs(pms_result[0] - mp_qext) / max(mp_qext, 1e-15),
                "Qsca": abs(pms_result[1] - mp_qsca) / max(mp_qsca, 1e-15),
            }
        }
    
    # ── Private Helpers ──────────────────────────────────────────────────
    
    def _single_angle_intensity(
        self,
        diameter_um: float,
        wavelength_um: float,
        m: complex,
        angle_deg: float
    ) -> float:
        """
        Compute unpolarised scattering intensity at a single angle.
        
        Uses miepython's i_unpolarized for speed at single angles.
        Returns differential scattering cross-section in µm²/sr.
        """
        x = np.pi * diameter_um / wavelength_um
        mu = np.cos(np.deg2rad(angle_deg))
        
        # miepython.i_unpolarized(m, x, mu) → intensity
        # This is |S1|² + |S2|² (unpolarised)
        intensity = miepython.i_unpolarized(m, x, mu)
        
        # Ensure scalar
        if hasattr(intensity, "__len__"):
            intensity = float(intensity[0]) if len(intensity) == 1 else float(np.sum(intensity))
        
        # Convert to differential cross-section (µm²/sr)
        # dσ/dΩ = (λ²/(4π²)) × (|S1|² + |S2|²) / 2
        prefactor = (wavelength_um ** 2) / (4.0 * np.pi ** 2)
        dsigma = prefactor * float(intensity) / 2.0
        
        return dsigma


def demo():
    """Quick demonstration of the Mie scattering engine."""
    engine = MieScatteringEngine()
    
    m = REFRACTIVE_INDICES["ammonium_sulfate"]
    
    print("=" * 70)
    print("MINI-LIDAR MIE SCATTERING ENGINE — DEMO")
    print("=" * 70)
    
    # Single particle
    print("\n[1] Single particle scattering (d = 1.0 µm, ammonium sulfate)")
    signals = engine.compute_scattering(1.0, m=m)
    print(f"    Scattering matrix (3λ × 3θ) [µm²/sr]:")
    print(f"    {'':>12s}  {'30°':>12s}  {'60°':>12s}  {'90°':>12s}")
    for i, wl in enumerate(WAVELENGTHS_NM):
        print(f"    {wl} nm:   {signals[i,0]:12.4e}  {signals[i,1]:12.4e}  {signals[i,2]:12.4e}")
    
    # Efficiencies
    print(f"\n[2] Mie efficiencies (d = 1.0 µm, λ = 532 nm)")
    eff = engine.compute_efficiencies(1.0, 0.532, m)
    for key, val in eff.items():
        print(f"    {key:>6s} = {val:.4f}")
    
    # Ångström exponent
    print(f"\n[3] Ångström exponent (450/650 nm ratio)")
    for d in [0.1, 0.5, 1.0, 2.0, 5.0]:
        alpha = engine.compute_angstrom_exponent(d, m)
        label = "fine" if alpha > 1.5 else ("coarse" if alpha < 0.5 else "mixed")
        print(f"    d = {d:5.2f} µm  →  α = {alpha:6.3f}  ({label})")
    
    # Cross-validation
    print(f"\n[4] PyMieScatt vs miepython cross-validation (d = 2.0 µm, λ = 532 nm)")
    cv = engine.cross_validate_miepython(2.0, 0.532, m)
    print(f"    PyMieScatt: Qext={cv['pymie']['Qext']:.6f}, Qsca={cv['pymie']['Qsca']:.6f}")
    print(f"    miepython:  Qext={cv['miepy']['Qext']:.6f}, Qsca={cv['miepy']['Qsca']:.6f}")
    print(f"    Rel. error: Qext={cv['relative_error']['Qext']:.2e}, Qsca={cv['relative_error']['Qsca']:.2e}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
