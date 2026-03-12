"""
Sensor Model
=============
Simulates the complete analog signal chain:
    Scattering → Photodiode → TIA → ADC → Digital value

Noise sources modelled:
    - Shot noise (Poisson statistics of photon detection)
    - Dark current noise
    - Thermal (Johnson-Nyquist) noise in TIA
    - ADC quantisation noise
    - Electronic 1/f noise (simplified)
"""

import numpy as np

from config import (
    LASER_POWER_MW, PHOTODIODE_AREA_MM2, TIA_GAIN_OHM,
    ADC_BITS, ADC_VREF, ADC_NOISE_LSB, DARK_CURRENT_NA,
    SCATTERING_VOLUME_MM3, WAVELENGTHS_NM, DETECTOR_ANGLES_DEG
)


class SensorModel:
    """
    Convert physical scattering cross-sections to realistic ADC readings
    with proper noise modelling.
    
    Signal chain:
        1. Scattering intensity → collected photon flux
        2. Photon flux → photocurrent (photodiode responsivity)
        3. Photocurrent → voltage (TIA gain)
        4. Voltage → ADC counts (quantisation)
        5. Add noise at each stage
    
    Usage:
        sensor = SensorModel()
        adc_readings, voltages = sensor.simulate_reading(scattering_matrix)
    """
    
    def __init__(self):
        self.laser_power_W = LASER_POWER_MW * 1e-3
        self.pd_area_m2 = PHOTODIODE_AREA_MM2 * 1e-6
        self.tia_gain = TIA_GAIN_OHM
        self.adc_bits = ADC_BITS
        self.adc_vref = ADC_VREF
        self.adc_lsb = ADC_VREF / (2 ** ADC_BITS)
        self.dark_current_A = DARK_CURRENT_NA * 1e-9
        
        # Photodiode responsivity (A/W) — typical for Si at each wavelength
        self.responsivity = {
            450: 0.25,   # lower at blue
            532: 0.35,   # good at green
            650: 0.42,   # best at red
        }
        
        # Detector solid angle (approximate, from collimator geometry)
        # Collimator: length 10mm, aperture 3mm → half-angle ≈ 8.5°
        self.detector_solid_angle_sr = 2 * np.pi * (1 - np.cos(np.deg2rad(8.5)))
        
        # Scattering volume (intersection of beam and detector FOV)
        self.scattering_volume_m3 = SCATTERING_VOLUME_MM3 * 1e-9
        
        # Bandwidth of detection (post-TIA filter cutoff)
        self.bandwidth_hz = 100.0
        
        # Boltzmann constant × temperature
        self.kT = 1.38e-23 * 300  # at 300K
        
    def scattering_to_photocurrent(
        self,
        dsigma_per_sr: float,
        number_density_per_m3: float,
        wavelength_nm: int
    ) -> float:
        """
        Convert differential scattering cross-section to photocurrent.
        
        I_sca = P_laser × n × V × (dσ/dΩ) × Ω_det × R
        
        Parameters
        ----------
        dsigma_per_sr : float
            Differential scattering cross-section (µm²/sr).
            From MieScatteringEngine.
        number_density_per_m3 : float
            Particle number concentration (#/m³).
        wavelength_nm : int
            Laser wavelength.
            
        Returns
        -------
        photocurrent_A : float
        """
        # Convert cross-section µm² → m²
        dsigma_m2 = dsigma_per_sr * 1e-12
        
        # Scattered power collected by detector
        P_scattered = (
            self.laser_power_W *
            number_density_per_m3 *
            self.scattering_volume_m3 *
            dsigma_m2 *
            self.detector_solid_angle_sr
        )
        
        # Photocurrent
        R = self.responsivity.get(wavelength_nm, 0.35)
        photocurrent = P_scattered * R
        
        return photocurrent
    
    def add_noise(
        self,
        photocurrent_A: float,
        wavelength_nm: int
    ) -> tuple:
        """
        Add realistic noise sources to the photocurrent.
        
        Returns
        -------
        (noisy_voltage, noise_components) : tuple
            noisy_voltage: TIA output with all noise sources.
            noise_components: dict of individual noise contributions.
        """
        # Signal voltage
        v_signal = photocurrent_A * self.tia_gain
        
        # 1. Shot noise (signal + dark current)
        total_current = photocurrent_A + self.dark_current_A
        shot_noise_rms = np.sqrt(2 * 1.6e-19 * total_current * self.bandwidth_hz)
        v_shot = shot_noise_rms * self.tia_gain * np.random.randn()
        
        # 2. Thermal noise in TIA feedback resistor
        thermal_noise_rms = np.sqrt(4 * self.kT * self.bandwidth_hz / self.tia_gain)
        v_thermal = thermal_noise_rms * self.tia_gain * np.random.randn()
        
        # 3. Dark current contribution
        v_dark = self.dark_current_A * self.tia_gain
        
        # 4. 1/f noise (simplified as fraction of signal)
        v_flicker = 0.002 * max(v_signal, 1e-6) * np.random.randn()
        
        # Total voltage
        v_total = v_signal + v_dark + v_shot + v_thermal + v_flicker
        
        noise_components = {
            "signal_V":   v_signal,
            "dark_V":     v_dark,
            "shot_V":     abs(v_shot),
            "thermal_V":  abs(v_thermal),
            "flicker_V":  abs(v_flicker),
            "total_V":    v_total,
            "snr_db":     10 * np.log10(v_signal ** 2 / max(v_shot ** 2 + v_thermal ** 2, 1e-30)),
        }
        
        return v_total, noise_components
    
    def voltage_to_adc(self, voltage: float) -> int:
        """
        Simulate 16-bit ADC conversion with quantisation noise.
        
        Returns ADC count (0 to 65535).
        """
        # Clamp to ADC range
        v_clamped = np.clip(voltage, 0, self.adc_vref)
        
        # Ideal conversion
        adc_ideal = v_clamped / self.adc_lsb
        
        # Add ADC quantisation noise
        adc_noisy = adc_ideal + ADC_NOISE_LSB * np.random.randn()
        
        # Quantise
        adc_count = int(np.clip(np.round(adc_noisy), 0, 2 ** self.adc_bits - 1))
        
        return adc_count
    
    def simulate_reading(
        self,
        scattering_matrix: np.ndarray,
        number_density_per_m3: float = 1e9
    ) -> tuple:
        """
        Full signal chain: scattering → ADC reading for all 9 channels.
        
        Parameters
        ----------
        scattering_matrix : np.ndarray, shape (3, 3)
            Differential scattering cross-sections [wavelength, angle].
            From MieScatteringEngine.compute_scattering_distribution().
        number_density_per_m3 : float
            Particle number density.
            
        Returns
        -------
        (adc_readings, voltages, snr_db) : tuple
            adc_readings: shape (3, 3) int array
            voltages: shape (3, 3) float array
            snr_db: shape (3, 3) float array
        """
        adc_readings = np.zeros((3, 3), dtype=int)
        voltages = np.zeros((3, 3))
        snr_db = np.zeros((3, 3))
        
        for i, wl_nm in enumerate(WAVELENGTHS_NM):
            for j, angle in enumerate(DETECTOR_ANGLES_DEG):
                # Physical scattering → photocurrent
                I_photo = self.scattering_to_photocurrent(
                    scattering_matrix[i, j],
                    number_density_per_m3,
                    wl_nm
                )
                
                # Add noise → voltage
                v_noisy, noise = self.add_noise(I_photo, wl_nm)
                
                # ADC conversion
                adc = self.voltage_to_adc(v_noisy)
                
                adc_readings[i, j] = adc
                voltages[i, j] = v_noisy
                snr_db[i, j] = noise["snr_db"]
        
        return adc_readings, voltages, snr_db

    def simulate_reading_fast(
        self,
        scattering_matrix: np.ndarray,
        number_density_per_m3: float = 1e9
    ) -> np.ndarray:
        """
        Fast version returning only ADC voltages (for ML training).
        Skips ADC quantisation for floating-point features.
        """
        voltages = np.zeros((3, 3))
        
        for i, wl_nm in enumerate(WAVELENGTHS_NM):
            for j in range(3):
                I_photo = self.scattering_to_photocurrent(
                    scattering_matrix[i, j],
                    number_density_per_m3,
                    wl_nm
                )
                v_noisy, _ = self.add_noise(I_photo, wl_nm)
                voltages[i, j] = max(v_noisy, 0)
        
        return voltages


def demo():
    """Demonstrate the sensor model."""
    sensor = SensorModel()
    
    print("=" * 70)
    print("SENSOR ELECTRONICS MODEL — DEMO")
    print("=" * 70)
    
    # Fake scattering matrix (typical values)
    scattering = np.array([
        [1e-2, 5e-3, 2e-3],    # 450 nm at 30°, 60°, 90°
        [8e-3, 4e-3, 1.5e-3],  # 532 nm
        [6e-3, 3e-3, 1e-3],    # 650 nm
    ])
    
    n_density = 1e9  # #/m³
    
    adc, volts, snr = sensor.simulate_reading(scattering, n_density)
    
    print(f"\n  Number density: {n_density:.0e} #/m³")
    print(f"\n  ADC Readings (16-bit):")
    print(f"  {'':>12s}  {'30°':>8s}  {'60°':>8s}  {'90°':>8s}")
    for i, wl in enumerate(WAVELENGTHS_NM):
        print(f"  {wl} nm:   {adc[i,0]:8d}  {adc[i,1]:8d}  {adc[i,2]:8d}")
    
    print(f"\n  Voltages (mV):")
    for i, wl in enumerate(WAVELENGTHS_NM):
        print(f"  {wl} nm:   {volts[i,0]*1e3:8.3f}  {volts[i,1]*1e3:8.3f}  {volts[i,2]*1e3:8.3f}")
    
    print(f"\n  SNR (dB):")
    for i, wl in enumerate(WAVELENGTHS_NM):
        print(f"  {wl} nm:   {snr[i,0]:8.1f}  {snr[i,1]:8.1f}  {snr[i,2]:8.1f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
