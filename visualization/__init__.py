"""
Visualization Package
=====================
Publication-quality plots for the Mini-LIDAR simulation framework.

Modules:
    style           — Shared matplotlib configuration, palettes, helpers
    plot_aerosol    — Aerosol size distribution plots
    plot_mie        — Mie efficiencies, phase functions, Angstrom exponent
    plot_hygroscopic — Hygroscopic growth and scattering enhancement
    plot_sensor     — Sensor signal heatmap, channel correlations
    plot_ml         — ML scatter plots, feature importance, ablation, humidity
"""

from visualization.plot_aerosol import plot_size_distribution
from visualization.plot_mie import (
    plot_mie_efficiencies,
    plot_phase_functions,
    plot_angstrom_analysis,
)
from visualization.plot_hygroscopic import plot_hygroscopic_growth
from visualization.plot_sensor import plot_sensor_heatmap
from visualization.plot_ml import (
    plot_ml_results,
    plot_feature_importance,
    plot_ablation_study,
    plot_humidity_analysis,
    plot_model_comparison_dashboard,
)

__all__ = [
    "plot_size_distribution",
    "plot_mie_efficiencies",
    "plot_phase_functions",
    "plot_angstrom_analysis",
    "plot_hygroscopic_growth",
    "plot_sensor_heatmap",
    "plot_ml_results",
    "plot_feature_importance",
    "plot_ablation_study",
    "plot_humidity_analysis",
    "plot_model_comparison_dashboard",
]
