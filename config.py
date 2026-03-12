import numpy as np

WAVELENGTHS_NM = [450, 532, 650]          # blue, green, red
WAVELENGTHS_UM = [w / 1000 for w in WAVELENGTHS_NM]

# ── Detector Angles (degrees)
DETECTOR_ANGLES_DEG = [30, 60, 90]        # forward, intermediate, side
DETECTOR_ANGLES_RAD = [np.deg2rad(a) for a in DETECTOR_ANGLES_DEG]

# ── Aerosol Physical Parameters
# Refractive indices at ~550 nm (real, imaginary)
REFRACTIVE_INDICES = {
    "ammonium_sulfate": complex(1.53, 0.0),
    "black_carbon":     complex(1.95, 0.79),
    "mineral_dust":     complex(1.53, 0.008),
    "organic_carbon":   complex(1.55, 0.03),
    "sea_salt":         complex(1.54, 0.0),
    "water":            complex(1.33, 0.0),
}

# Lognormal size distribution parameters for common aerosol types
# (geometric_mean_diameter_um, geometric_standard_deviation)
AEROSOL_MODES = {
    "urban_fine":       {"Dg": 0.15,  "sigma_g": 1.8,  "N_frac": 0.70},
    "urban_coarse":     {"Dg": 3.5,   "sigma_g": 2.2,  "N_frac": 0.05},
    "accumulation":     {"Dg": 0.5,   "sigma_g": 2.0,  "N_frac": 0.25},
    "biomass_fine":     {"Dg": 0.13,  "sigma_g": 1.7,  "N_frac": 0.80},
    "biomass_coarse":   {"Dg": 1.0,   "sigma_g": 2.0,  "N_frac": 0.20},
    "dust_fine":        {"Dg": 0.8,   "sigma_g": 2.0,  "N_frac": 0.30},
    "dust_coarse":      {"Dg": 5.0,   "sigma_g": 2.5,  "N_frac": 0.70},
}

# ── Hygroscopic Growth Parameters (kappa-Köhler)
KAPPA_VALUES = {
    "ammonium_sulfate": 0.53,
    "black_carbon":     0.0,
    "mineral_dust":     0.01,
    "organic_carbon":   0.10,
    "sea_salt":         1.12,
}

# ── Sensor / Electronics Parameters 
LASER_POWER_MW       = 5.0          # per channel
PHOTODIODE_AREA_MM2  = 7.0          # BPW34 active area
TIA_GAIN_OHM         = 1e7          # transimpedance gain (10 MΩ)
ADC_BITS             = 16           # ADS1115
ADC_VREF             = 4.096        # PGA full-scale voltage
ADC_NOISE_LSB        = 2.0          # RMS noise in LSB
DARK_CURRENT_NA      = 2.0          # photodiode dark current
SCATTERING_VOLUME_MM3 = 10.0        # approximate
FLOW_RATE_LPM        = 0.3          # litres per minute

# ── Environmental Ranges
TEMP_RANGE_C  = (5.0, 45.0)
RH_RANGE_PCT  = (15.0, 95.0)

# ── Simulation Parameters
N_DIAMETER_BINS      = 200          # size distribution resolution
DIAMETER_RANGE_UM    = (0.01, 20.0) # particle diameter range
N_SAMPLES            = 5000         # synthetic training samples
PM_CONC_RANGE        = (5.0, 500.0) # PM concentration range (µg/m³)
PARTICLE_DENSITY     = 1.5e-12      # g/µm³ (1.5 g/cm³)

# ── ML Parameters 
TEST_SIZE             = 0.20
RANDOM_STATE          = 42
RF_N_ESTIMATORS       = 200
RF_MAX_DEPTH          = 15
XGB_N_ESTIMATORS      = 300
XGB_LEARNING_RATE     = 0.05
XGB_MAX_DEPTH         = 8
NN_HIDDEN_LAYERS      = [64, 32]
NN_EPOCHS             = 200
NN_BATCH_SIZE         = 64
