"""
Plot Style Configuration
========================
Shared matplotlib settings, colour palettes, and helper utilities
used across all visualization modules.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for WSL / headless servers
import matplotlib.pyplot as plt
import os


# ── Global Style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "figure.facecolor":   "white",
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
    "legend.frameon":     False,
    "legend.fontsize":    9,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})

# ── Colour Palettes ─────────────────────────────────────────────────────
COLORS = {
    "primary":   ["#2E86AB", "#A23B72", "#F18F01", "#3B9979", "#C73E1D"],
    "wavelength": {450: "#2E86AB", 532: "#3B9979", 650: "#C73E1D"},
    "models":    {"RandomForest": "#2E86AB", "XGBoost": "#F18F01",
                  "GradientBoosting": "#F18F01", "NeuralNetwork": "#A23B72"},
    "aerosol":   {"ammonium_sulfate": "#2E86AB", "organic_carbon": "#3B9979",
                  "sea_salt": "#F18F01", "black_carbon": "#333333",
                  "mineral_dust": "#C73E1D"},
    "scenarios": {"urban": "#2E86AB", "biomass": "#A23B72", "dust": "#F18F01",
                  "clean": "#3B9979", "mixed": "#C73E1D"},
}


def save_figure(fig, output_path: str):
    """Save figure with automatic directory creation."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"    Saved: {output_path}")
