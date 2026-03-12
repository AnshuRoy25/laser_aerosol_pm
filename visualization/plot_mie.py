"""
Plot: Mie Scattering Physics
==============================
Scattering/extinction efficiencies vs particle diameter,
and angular phase functions for key particle sizes.
"""

import numpy as np
import matplotlib.pyplot as plt

from visualization.style import COLORS, save_figure


def plot_mie_efficiencies(
    diameters: np.ndarray,
    efficiencies: dict,
    output_path: str = "results/mie_efficiencies.png"
):
    """
    Plot Qsca, Qext, and Qsca×πr² vs diameter for each wavelength.

    Parameters
    ----------
    diameters : array
        Particle diameters (µm).
    efficiencies : dict
        {wavelength_nm: {"Qsca": array, "Qext": array}}
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    wl_colors = COLORS["wavelength"]

    # ── Panel 1: Scattering efficiency ───────────────────────────────
    ax = axes[0]
    for wl_nm, data in efficiencies.items():
        ax.plot(diameters, data["Qsca"], color=wl_colors[wl_nm],
                linewidth=2, label=f"{wl_nm} nm")
    ax.set_xscale("log")
    ax.set_xlabel("Particle Diameter (µm)")
    ax.set_ylabel("Q$_{sca}$")
    ax.set_title("Scattering Efficiency")
    ax.legend()
    ax.grid(True)

    # ── Panel 2: Extinction efficiency ───────────────────────────────
    ax = axes[1]
    for wl_nm, data in efficiencies.items():
        ax.plot(diameters, data["Qext"], color=wl_colors[wl_nm],
                linewidth=2, label=f"{wl_nm} nm")
    ax.set_xscale("log")
    ax.set_xlabel("Particle Diameter (µm)")
    ax.set_ylabel("Q$_{ext}$")
    ax.set_title("Extinction Efficiency")
    ax.legend()
    ax.grid(True)

    # ── Panel 3: Scattering cross-section σ_sca = Qsca × π(D/2)² ───
    ax = axes[2]
    for wl_nm, data in efficiencies.items():
        sigma_sca = data["Qsca"] * np.pi * (diameters / 2) ** 2  # µm²
        ax.plot(diameters, sigma_sca, color=wl_colors[wl_nm],
                linewidth=2, label=f"{wl_nm} nm")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Particle Diameter (µm)")
    ax.set_ylabel("σ$_{sca}$  (µm²)")
    ax.set_title("Scattering Cross-Section")
    ax.legend()
    ax.grid(True, which="both")

    fig.suptitle("Mie Theory — Optical Properties vs Particle Size",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_phase_functions(
    angles: np.ndarray,
    phase_data: dict,
    output_path: str = "results/phase_functions.png"
):
    """
    Plot scattering phase functions (Cartesian + Polar).

    Parameters
    ----------
    phase_data : dict
        {diameter_label: {"theta": array, "SU": array}}
    """
    fig = plt.figure(figsize=(15, 6))
    ax_cart = fig.add_subplot(121)
    ax_polar = fig.add_subplot(122, projection="polar")

    palette = COLORS["primary"]

    # ── Cartesian ────────────────────────────────────────────────────
    for i, (label, data) in enumerate(phase_data.items()):
        ax_cart.semilogy(data["theta"], data["SU"],
                         color=palette[i % len(palette)],
                         linewidth=2, label=label)

    # Mark Mini-LIDAR detector angles
    for angle in [30, 60, 90]:
        ax_cart.axvline(angle, color="#999999", linestyle="--",
                        linewidth=0.8, alpha=0.6)
        ax_cart.annotate(f"{angle}°", xy=(angle, 0),
                         xytext=(angle + 2, ax_cart.get_ylim()[1] * 0.3),
                         fontsize=8, color="#666666")

    ax_cart.set_xlabel("Scattering Angle θ (°)")
    ax_cart.set_ylabel("Scattering Intensity (a.u.)")
    ax_cart.set_title("Phase Function — Cartesian")
    ax_cart.legend(fontsize=8)
    ax_cart.grid(True)
    ax_cart.set_xlim(0, 180)

    # ── Polar ────────────────────────────────────────────────────────
    for i, (label, data) in enumerate(phase_data.items()):
        theta_rad = np.deg2rad(data["theta"])
        su_norm = np.log10(data["SU"] / data["SU"].max() + 1e-10) + 10
        su_norm = np.clip(su_norm, 0, None)
        ax_polar.plot(theta_rad, su_norm, color=palette[i % len(palette)],
                      linewidth=1.5, label=label)

    # Mark detector angles on polar
    for angle in [30, 60, 90]:
        ax_polar.axvline(np.deg2rad(angle), color="#999999",
                         linestyle="--", linewidth=0.5, alpha=0.5)

    ax_polar.set_title("Phase Function — Polar (log scale)", pad=20)
    ax_polar.legend(fontsize=7, loc="upper right",
                    bbox_to_anchor=(1.35, 1.05))

    fig.suptitle("Scattering Phase Functions at λ = 532 nm",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_angstrom_analysis(
    diameters: np.ndarray,
    angstrom_values: np.ndarray,
    output_path: str = "results/angstrom_exponent.png"
):
    """
    Plot Ångström exponent vs particle diameter with size-regime
    classification bands.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Classification shading
    ax.axhspan(1.5, 4.5, alpha=0.08, color="#2E86AB",
               label="Fine-dominated (α > 1.5)")
    ax.axhspan(0.5, 1.5, alpha=0.08, color="#3B9979",
               label="Mixed (0.5 < α < 1.5)")
    ax.axhspan(-2.0, 0.5, alpha=0.08, color="#F18F01",
               label="Coarse-dominated (α < 0.5)")

    ax.plot(diameters, angstrom_values, color=COLORS["primary"][0],
            linewidth=2.5, zorder=5)

    # Reference lines
    ax.axhline(y=0, color="#999999", linewidth=0.5, linestyle="-")
    ax.axhline(y=1.5, color="#2E86AB", linewidth=0.5, linestyle=":")
    ax.axhline(y=0.5, color="#F18F01", linewidth=0.5, linestyle=":")

    ax.set_xscale("log")
    ax.set_xlabel("Particle Diameter (µm)")
    ax.set_ylabel("Ångström Exponent  α  (450 / 650 nm)")
    ax.set_title("Ångström Exponent — Particle Size Indicator")
    ax.legend(fontsize=9)
    ax.grid(True)
    ax.set_xlim(0.05, 15)
    ax.set_ylim(-2, 4.5)

    fig.tight_layout()
    save_figure(fig, output_path)
