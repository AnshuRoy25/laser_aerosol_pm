"""
Aerosol Generator
=================
Generates realistic aerosol size distributions using lognormal modes.
Supports urban, biomass burning, dust, and mixed scenarios common
in Indian environments (relevant to Himachal Pradesh deployment).

Physics:
    - Multi-modal lognormal distributions: dN/dlogD = Σ Nᵢ/(√(2π)·ln(σᵢ)) · exp(-(ln(D)-ln(Dᵢ))²/(2·ln²(σᵢ)))
    - PM2.5/PM10 mass from volume integration with density
    - Realistic number concentration scaling
"""

import numpy as np
from scipy.stats import lognorm

from config import (
    AEROSOL_MODES, N_DIAMETER_BINS, DIAMETER_RANGE_UM,
    PM_CONC_RANGE, PARTICLE_DENSITY
)


class AerosolGenerator:
    """
    Generate synthetic aerosol size distributions and corresponding
    PM2.5/PM10 mass concentrations.
    
    Usage:
        gen = AerosolGenerator()
        diameters, dNdlogD, pm25, pm10 = gen.generate_sample(
            pm25_target=35.0, aerosol_scenario="urban"
        )
    """
    
    def __init__(self, n_bins: int = N_DIAMETER_BINS):
        self.n_bins = n_bins
        self.d_min, self.d_max = DIAMETER_RANGE_UM
        
        # Logarithmically spaced diameter bins
        self.diameters = np.logspace(
            np.log10(self.d_min), np.log10(self.d_max), n_bins
        )
        self.log_diameters = np.log10(self.diameters)
        self.dlogD = np.diff(self.log_diameters)
        self.dlogD = np.append(self.dlogD, self.dlogD[-1])  # pad last bin
        
        # Scenarios: combinations of modes that represent real conditions
        self.scenarios = {
            "urban": [
                {"mode": "urban_fine",   "weight": 0.70},
                {"mode": "accumulation", "weight": 0.25},
                {"mode": "urban_coarse", "weight": 0.05},
            ],
            "biomass": [
                {"mode": "biomass_fine",   "weight": 0.80},
                {"mode": "biomass_coarse", "weight": 0.20},
            ],
            "dust": [
                {"mode": "dust_fine",   "weight": 0.30},
                {"mode": "dust_coarse", "weight": 0.70},
            ],
            "clean": [
                {"mode": "urban_fine",   "weight": 0.60},
                {"mode": "accumulation", "weight": 0.40},
            ],
            "mixed": [
                {"mode": "urban_fine",     "weight": 0.40},
                {"mode": "accumulation",   "weight": 0.20},
                {"mode": "biomass_fine",   "weight": 0.20},
                {"mode": "dust_coarse",    "weight": 0.20},
            ],
        }
    
    def lognormal_dNdlogD(
        self,
        Dg: float,
        sigma_g: float,
        N_total: float
    ) -> np.ndarray:
        """
        Compute lognormal number distribution dN/dlogD.
        
        Parameters
        ----------
        Dg : float
            Geometric mean diameter (µm).
        sigma_g : float
            Geometric standard deviation.
        N_total : float
            Total number concentration (#/cm³).
            
        Returns
        -------
        dNdlogD : array
            Number distribution at self.diameters.
        """
        ln_sigma = np.log(sigma_g)
        ln_D = np.log(self.diameters)
        ln_Dg = np.log(Dg)
        
        dNdlogD = (N_total / (np.sqrt(2 * np.pi) * ln_sigma)) * \
                  np.exp(-(ln_D - ln_Dg) ** 2 / (2 * ln_sigma ** 2))
        
        return dNdlogD
    
    def compute_pm_mass(
        self,
        dNdlogD: np.ndarray,
        cutoff_um: float = 2.5,
        density: float = PARTICLE_DENSITY
    ) -> float:
        """
        Compute PM mass concentration from size distribution.
        
        PM_x = ∫₀ˣ (π/6)·D³·ρ·(dN/dlogD)·dlogD
        
        Parameters
        ----------
        dNdlogD : array
            Number distribution.
        cutoff_um : float
            Upper diameter cutoff (2.5 for PM2.5, 10 for PM10).
        density : float
            Particle density in g/µm³.
            
        Returns
        -------
        pm_mass : float
            Mass concentration in µg/m³.
        """
        mask = self.diameters <= cutoff_um
        volumes = (np.pi / 6.0) * self.diameters[mask] ** 3   # µm³
        masses = volumes * density                              # g per particle
        
        # dN/dlogD × dlogD gives number per cm³ per bin
        dN = dNdlogD[mask] * self.dlogD[mask]
        
        # Total mass: g/cm³ → µg/m³ (multiply by 1e6 × 1e6 = 1e12)
        pm_mass = np.sum(masses * dN) * 1e12
        
        return pm_mass
    
    def generate_distribution(
        self,
        scenario: str = "urban",
        N_total: float = 10000.0,
        perturbation: float = 0.15
    ) -> np.ndarray:
        """
        Generate a multi-modal size distribution for a given scenario.
        
        Parameters
        ----------
        scenario : str
            One of: urban, biomass, dust, clean, mixed.
        N_total : float
            Total number concentration (#/cm³).
        perturbation : float
            Random variation factor (0 = exact, 0.15 = ±15%).
            
        Returns
        -------
        dNdlogD : array
            Combined number distribution.
        """
        modes = self.scenarios[scenario]
        dNdlogD = np.zeros_like(self.diameters)
        
        for mode_spec in modes:
            mode_params = AEROSOL_MODES[mode_spec["mode"]]
            weight = mode_spec["weight"]
            
            # Add random perturbation for realism
            Dg = mode_params["Dg"] * (1 + perturbation * np.random.randn())
            Dg = max(Dg, 0.01)  # keep physical
            sigma_g = mode_params["sigma_g"] * (1 + 0.05 * np.random.randn())
            sigma_g = max(sigma_g, 1.1)
            
            mode_N = N_total * weight * (1 + perturbation * np.random.randn())
            mode_N = max(mode_N, 0)
            
            dNdlogD += self.lognormal_dNdlogD(Dg, sigma_g, mode_N)
        
        return dNdlogD
    
    def generate_sample(
        self,
        pm25_target: float = None,
        scenario: str = "urban",
        perturbation: float = 0.15
    ) -> tuple:
        """
        Generate a single sample with target PM2.5 concentration.
        
        Iteratively scales the number concentration to match the
        target PM2.5 within ~5% tolerance.
        
        Parameters
        ----------
        pm25_target : float, optional
            Desired PM2.5 (µg/m³). If None, random from PM_CONC_RANGE.
        scenario : str
            Aerosol scenario type.
            
        Returns
        -------
        (diameters, dNdlogD, pm25, pm10) : tuple
        """
        if pm25_target is None:
            pm25_target = np.random.uniform(*PM_CONC_RANGE)
        
        # Start with initial guess for N_total
        N_total = 5000.0
        dNdlogD = self.generate_distribution(scenario, N_total, perturbation)
        
        # Scale to match target PM2.5
        pm25_initial = self.compute_pm_mass(dNdlogD, cutoff_um=2.5)
        
        if pm25_initial > 0:
            scale = pm25_target / pm25_initial
            dNdlogD *= scale
        
        pm25 = self.compute_pm_mass(dNdlogD, cutoff_um=2.5)
        pm10 = self.compute_pm_mass(dNdlogD, cutoff_um=10.0)
        
        return self.diameters, dNdlogD, pm25, pm10
    
    def generate_dataset(
        self,
        n_samples: int,
        scenarios: list = None,
        pm_range: tuple = PM_CONC_RANGE
    ) -> list:
        """
        Generate a full dataset of samples spanning conditions.
        
        Returns
        -------
        samples : list of dict
            Each dict: {diameters, dNdlogD, pm25, pm10, scenario, temperature, rh}
        """
        if scenarios is None:
            scenarios = list(self.scenarios.keys())
        
        samples = []
        
        for i in range(n_samples):
            scenario = np.random.choice(scenarios)
            pm25_target = np.random.uniform(*pm_range)
            
            # Random environmental conditions
            temperature = np.random.uniform(5, 45)
            rh = np.random.uniform(15, 95)
            
            diameters, dNdlogD, pm25, pm10 = self.generate_sample(
                pm25_target=pm25_target,
                scenario=scenario
            )
            
            samples.append({
                "diameters":   diameters,
                "dNdlogD":     dNdlogD,
                "pm25":        pm25,
                "pm10":        pm10,
                "scenario":    scenario,
                "temperature": temperature,
                "rh":          rh,
            })
            
        return samples


def demo():
    """Demonstrate the aerosol generator."""
    gen = AerosolGenerator()
    
    print("=" * 70)
    print("AEROSOL SIZE DISTRIBUTION GENERATOR — DEMO")
    print("=" * 70)
    
    for scenario in ["urban", "biomass", "dust", "clean", "mixed"]:
        diameters, dNdlogD, pm25, pm10 = gen.generate_sample(
            pm25_target=50.0, scenario=scenario
        )
        print(f"\n  {scenario:>10s}:  PM2.5 = {pm25:7.1f} µg/m³  |  PM10 = {pm10:7.1f} µg/m³  |  PM2.5/PM10 = {pm25/pm10:.2f}")
    
    print(f"\n  Generating {100} random samples...")
    dataset = gen.generate_dataset(100)
    pm25_vals = [s["pm25"] for s in dataset]
    print(f"  PM2.5 range: {min(pm25_vals):.1f} — {max(pm25_vals):.1f} µg/m³")
    print(f"  PM2.5 mean:  {np.mean(pm25_vals):.1f} µg/m³")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
