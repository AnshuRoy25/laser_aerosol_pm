"""
Plot: Hygroscopic Growth
=========================
Growth factors, scattering enhancement f(RH), and
core-shell refractive index change with humidity.
"""

import numpy as np
import matplotlib.pyplot as plt

from visualization.style import COLORS, save_figure


def plot_hygroscopic_growth(
    rh_values: np.ndarray,
    growth_data: dict,
    output_path: str = "results/hygroscopic_growth.png"
):
    """
    Three-panel plot: growth factor, scattering enhancement,
    and sensor bias illustration.

    Parameters
    ----------
    rh_values : array
        Relative humidity values (%).
    growth_data : dict
        {aerosol_type: {"growth_factor": array, "scattering_enhancement": array}}
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    palette = COLORS["aerosol"]
    fallback = COLORS["primary"]

    # ── Panel 1: Growth factor g(RH) ────────────────────────────────
    ax = axes[0]
    for i, (atype, data) in enumerate(growth_data.items()):
        c = palette.get(atype, fallback[i % len(fallback)])
        label = atype.replace("_", " ").title()
        ax.plot(rh_values, data["growth_factor"], color=c,
                linewidth=2.2, label=label)

    ax.axhline(y=1.0, color="#cccccc", linewidth=0.8)
    ax.set_xlabel("Relative Humidity (%)")
    ax.set_ylabel("Growth Factor  g(RH) = D$_{wet}$ / D$_{dry}$")
    ax.set_title("Hygroscopic Growth Factor")
    ax.legend(fontsize=8)
    ax.grid(True)
    ax.set_xlim(rh_values.min(), rh_values.max())

    # ── Panel 2: Scattering enhancement f(RH) ───────────────────────
    ax = axes[1]
    for i, (atype, data) in enumerate(growth_data.items()):
        if "scattering_enhancement" not in data:
            continue
        c = palette.get(atype, fallback[i % len(fallback)])
        label = atype.replace("_", " ").title()
        ax.plot(rh_values, data["scattering_enhancement"], color=c,
                linewidth=2.2, label=label)

    ax.axhline(y=1.0, color="#cccccc", linewidth=0.8)
    # Highlight danger zone
    ax.axhspan(2.0, ax.get_ylim()[1] if ax.get_ylim()[1] > 3 else 10,
               alpha=0.05, color="red")
    ax.text(75, 2.2, "Sensor bias zone", fontsize=8, color="#C73E1D",
            style="italic")

    ax.set_xlabel("Relative Humidity (%)")
    ax.set_ylabel("f(RH) = σ$_{sca,wet}$ / σ$_{sca,dry}$")
    ax.set_title("Scattering Enhancement Factor")
    ax.legend(fontsize=8)
    ax.grid(True)
    ax.set_xlim(rh_values.min(), rh_values.max())

    # ── Panel 3: Uncorrected vs corrected PM reading ─────────────────
    ax = axes[2]
    # Simulate what an uncorrected sensor would read
    if "ammonium_sulfate" in growth_data and "scattering_enhancement" in growth_data["ammonium_sulfate"]:
        f_rh = growth_data["ammonium_sulfate"]["scattering_enhancement"]
        true_pm = 50.0  # constant true PM2.5
        uncorrected = true_pm * f_rh
        corrected = np.full_like(rh_values, true_pm, dtype=float)

        ax.plot(rh_values, uncorrected, color="#C73E1D", linewidth=2.2,
                label="Uncorrected sensor", linestyle="--")
        ax.plot(rh_values, corrected, color="#3B9979", linewidth=2.2,
                label="ML-corrected sensor")
        ax.axhline(y=true_pm, color="#333333", linewidth=1,
                    linestyle=":", alpha=0.5, label="True PM2.5")

        ax.fill_between(rh_values, corrected, uncorrected,
                        alpha=0.12, color="#C73E1D")
        ax.annotate("Humidity bias\n(eliminated by ML)",
                    xy=(80, (uncorrected[np.argmin(np.abs(rh_values - 80))] + true_pm) / 2),
                    fontsize=8, color="#C73E1D", ha="center",
                    style="italic")

    ax.set_xlabel("Relative Humidity (%)")
    ax.set_ylabel("Reported PM2.5 (µg/m³)")
    ax.set_title("Humidity Bias Correction")
    ax.legend(fontsize=8)
    ax.grid(True)
    ax.set_xlim(rh_values.min(), rh_values.max())

    fig.suptitle("Humidity Effects on Aerosol Scattering & Sensor Reading",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, output_path)
