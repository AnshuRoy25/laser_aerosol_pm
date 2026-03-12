"""
Plot: Aerosol Size Distributions
=================================
Multi-modal lognormal distributions for different aerosol scenarios.
Shows PM2.5 and PM10 cutoff markers.
"""

import numpy as np
import matplotlib.pyplot as plt

from visualization.style import COLORS, save_figure


def plot_size_distribution(
    diameters: np.ndarray,
    distributions: dict,
    output_path: str = "results/size_distributions.png"
):
    """
    Plot aerosol size distributions for different scenarios.

    Parameters
    ----------
    diameters : array
        Diameter bin centres (µm).
    distributions : dict
        {scenario_name: dNdlogD_array}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    palette = COLORS["scenarios"]
    fallback = COLORS["primary"]

    # ── Left: number distribution ────────────────────────────────────
    ax = axes[0]
    for i, (name, dNdlogD) in enumerate(distributions.items()):
        c = palette.get(name, fallback[i % len(fallback)])
        ax.plot(diameters, dNdlogD, color=c, linewidth=2.2,
                label=name.capitalize())

    ax.axvline(2.5, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(10.0, color="#888888", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(2.7, ax.get_ylim()[0] * 5 if ax.get_ylim()[0] > 0 else 1,
            "PM2.5", fontsize=8, color="#666666")
    ax.text(10.5, ax.get_ylim()[0] * 5 if ax.get_ylim()[0] > 0 else 1,
            "PM10", fontsize=8, color="#666666")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Particle Diameter (µm)")
    ax.set_ylabel("dN/dlogD  (#/cm³)")
    ax.set_title("Number Size Distribution")
    ax.legend()
    ax.set_xlim(0.01, 20)
    ax.grid(True, which="both")

    # ── Right: volume (mass-proxy) distribution ──────────────────────
    ax2 = axes[1]
    for i, (name, dNdlogD) in enumerate(distributions.items()):
        c = palette.get(name, fallback[i % len(fallback)])
        # dV/dlogD ∝ D³ × dN/dlogD
        dVdlogD = (np.pi / 6.0) * diameters ** 3 * dNdlogD
        ax2.plot(diameters, dVdlogD, color=c, linewidth=2.2,
                 label=name.capitalize())

    ax2.axvline(2.5, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax2.axvline(10.0, color="#888888", linestyle=":", linewidth=1, alpha=0.7)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Particle Diameter (µm)")
    ax2.set_ylabel("dV/dlogD  (µm³/cm³)")
    ax2.set_title("Volume Size Distribution (Mass Proxy)")
    ax2.legend()
    ax2.set_xlim(0.01, 20)
    ax2.grid(True, which="both")

    fig.suptitle("Aerosol Size Distributions — Multi-Modal Lognormal",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_path)
