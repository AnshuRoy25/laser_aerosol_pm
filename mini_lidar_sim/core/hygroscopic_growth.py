"""
Hygroscopic Growth Model
========================
Models particle swelling due to water uptake using κ-Köhler theory.
This is THE critical correction for the Mini-LIDAR — humidity is the
single largest source of error in low-cost PM sensors.

Physics:
    - κ-Köhler: g(RH) = [1 + κ·(aw/(1-aw))]^(1/3)
    - Water activity aw ≈ RH/100 (bulk approximation, valid for D > 100 nm)
    - Core-shell model: dry core + water shell → modified refractive index
"""

import numpy as np

# Compatibility patch for scipy ≥ 1.14
import scipy.integrate
if not hasattr(scipy.integrate, "trapz"):
    scipy.integrate.trapz = scipy.integrate.trapezoid

from config import KAPPA_VALUES, REFRACTIVE_INDICES


class HygroscopicGrowthModel:
    """
    Compute hygroscopic growth factors and modified particle properties.
    
    Usage:
        hgm = HygroscopicGrowthModel()
        d_wet, m_wet = hgm.grow_particle(
            d_dry=0.5, rh=80.0, aerosol_type="ammonium_sulfate"
        )
    """
    
    def __init__(self):
        self.kappa = KAPPA_VALUES
        self.m_water = REFRACTIVE_INDICES["water"]
    
    def growth_factor(
        self,
        rh: float,
        kappa: float = None,
        aerosol_type: str = "ammonium_sulfate"
    ) -> float:
        """
        Compute hygroscopic growth factor g = D_wet / D_dry.
        
        Uses κ-Köhler: g³ = 1 + κ · aw/(1 - aw)
        where aw ≈ RH/100.
        
        Parameters
        ----------
        rh : float
            Relative humidity in percent (0-100).
        kappa : float, optional
            Hygroscopicity parameter. If None, uses aerosol_type.
        aerosol_type : str
            Fallback for kappa lookup.
            
        Returns
        -------
        g : float
            Growth factor (≥ 1.0).
        """
        if kappa is None:
            kappa = self.kappa.get(aerosol_type, 0.0)
        
        if kappa <= 0 or rh <= 0:
            return 1.0
        
        # Water activity (clamp to avoid division by zero)
        aw = min(rh / 100.0, 0.999)
        
        # κ-Köhler equation
        g_cubed = 1.0 + kappa * aw / (1.0 - aw)
        g = g_cubed ** (1.0 / 3.0)
        
        return max(g, 1.0)
    
    def grow_particle(
        self,
        d_dry: float,
        rh: float,
        kappa: float = None,
        aerosol_type: str = "ammonium_sulfate"
    ) -> tuple:
        """
        Compute wet particle diameter and effective refractive index.
        
        Core-shell volume mixing for refractive index:
            m_eff = f_core · m_dry + (1 - f_core) · m_water
        where f_core = (D_dry/D_wet)³ = 1/g³.
        
        Parameters
        ----------
        d_dry : float
            Dry particle diameter (µm).
        rh : float
            Relative humidity (%).
            
        Returns
        -------
        (d_wet, m_effective) : tuple
            Wet diameter (µm) and volume-weighted refractive index.
        """
        if kappa is None:
            kappa = self.kappa.get(aerosol_type, 0.0)
        
        m_dry = REFRACTIVE_INDICES.get(aerosol_type, complex(1.53, 0))
        
        g = self.growth_factor(rh, kappa=kappa)
        d_wet = d_dry * g
        
        # Volume fraction of dry core
        f_core = 1.0 / (g ** 3)
        
        # Volume-weighted mixing of refractive index
        m_eff = f_core * m_dry + (1.0 - f_core) * self.m_water
        
        return d_wet, m_eff
    
    def grow_distribution(
        self,
        diameters_dry: np.ndarray,
        dNdlogD: np.ndarray,
        rh: float,
        kappa: float = None,
        aerosol_type: str = "ammonium_sulfate"
    ) -> tuple:
        """
        Apply hygroscopic growth to an entire size distribution.
        
        Returns
        -------
        (diameters_wet, dNdlogD_wet, m_eff_array) : tuple
            Wet diameters, adjusted distribution, effective refractive indices.
        """
        g = self.growth_factor(rh, kappa=kappa, aerosol_type=aerosol_type)
        
        diameters_wet = diameters_dry * g
        
        # Number is conserved but bins shift → scale dN/dlogD
        # dN/dlogD_wet = dN/dlogD_dry (same number per log-bin)
        dNdlogD_wet = dNdlogD.copy()
        
        # Effective refractive index for each size
        m_dry = REFRACTIVE_INDICES.get(aerosol_type, complex(1.53, 0))
        f_core = 1.0 / (g ** 3)
        m_eff = f_core * m_dry + (1.0 - f_core) * self.m_water
        m_eff_array = np.full(len(diameters_dry), m_eff)
        
        return diameters_wet, dNdlogD_wet, m_eff_array
    
    def scattering_enhancement(
        self,
        d_dry: float,
        rh: float,
        wavelength_um: float = 0.532,
        aerosol_type: str = "ammonium_sulfate"
    ) -> float:
        """
        Compute f(RH) = σ_sca(wet) / σ_sca(dry) at a given wavelength.
        
        This is the key quantity that causes humidity bias in sensors.
        """
        import PyMieScatt as pms
        
        m_dry = REFRACTIVE_INDICES[aerosol_type]
        
        # Dry scattering
        res_dry = pms.MieQ(m_dry, wavelength_um * 1000, d_dry * 1000)
        sigma_dry = res_dry[1] * np.pi * (d_dry / 2) ** 2
        
        # Wet particle
        d_wet, m_wet = self.grow_particle(d_dry, rh, aerosol_type=aerosol_type)
        res_wet = pms.MieQ(m_wet, wavelength_um * 1000, d_wet * 1000)
        sigma_wet = res_wet[1] * np.pi * (d_wet / 2) ** 2
        
        if sigma_dry <= 0:
            return 1.0
            
        return sigma_wet / sigma_dry


def demo():
    """Demonstrate hygroscopic growth model."""
    hgm = HygroscopicGrowthModel()
    
    print("=" * 70)
    print("HYGROSCOPIC GROWTH MODEL — DEMO")
    print("=" * 70)
    
    print("\n[1] Growth factors for ammonium sulfate (κ = 0.53):")
    print(f"    {'RH%':>5s}  {'g(RH)':>8s}  {'D_wet/D_dry':>12s}  {'σ_wet/σ_dry':>12s}")
    for rh in [30, 50, 60, 70, 80, 85, 90, 95]:
        g = hgm.growth_factor(rh, aerosol_type="ammonium_sulfate")
        f_rh = hgm.scattering_enhancement(0.5, rh, aerosol_type="ammonium_sulfate")
        print(f"    {rh:5d}  {g:8.3f}  {g:12.3f}  {f_rh:12.2f}×")
    
    print(f"\n[2] Growth factors by aerosol type at RH = 80%:")
    for atype in ["ammonium_sulfate", "black_carbon", "mineral_dust", "organic_carbon", "sea_salt"]:
        kappa = KAPPA_VALUES[atype]
        g = hgm.growth_factor(80.0, aerosol_type=atype)
        print(f"    {atype:>20s}  κ = {kappa:.2f}  →  g(80%) = {g:.3f}")
    
    print("\n[3] Core-shell refractive index (d=0.5µm, ammonium sulfate):")
    for rh in [30, 60, 80, 90]:
        d_wet, m_eff = hgm.grow_particle(0.5, rh, aerosol_type="ammonium_sulfate")
        print(f"    RH={rh:3d}%:  d_wet = {d_wet:.3f} µm,  m_eff = {m_eff.real:.3f} + {m_eff.imag:.4f}i")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
