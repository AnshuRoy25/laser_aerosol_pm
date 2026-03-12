"""
Plot: Sensor Signal Analysis
==============================
Heatmap of 3λ × 3θ scattering channels, channel correlations,
and signal-to-noise analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from visualization.style import COLORS, save_figure


def plot_sensor_heatmap(
    X: np.ndarray,
    feature_names: list,
    output_path: str = "results/sensor_heatmap.png"
):
    """
    Heatmap of the 9 raw scattering channels across all samples,
    plus channel correlation matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (first 9 columns are raw channels).
    feature_names : list
        Feature names.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    raw_channels = X[:, :9]
    channel_names = feature_names[:9] if feature_names else \
        [f"Ch{i}" for i in range(9)]

    # ── Panel 1: Channel distributions (box plot) ────────────────────
    ax = axes[0]
    # Use log-transformed values for better visibility
    log_data = np.log10(np.clip(raw_channels, 1e-15, None))
    bp = ax.boxplot(log_data, labels=[n.replace("V_", "").replace("nm_", "\n")
                                       for n in channel_names],
                    patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        wl_idx = i // 3
        colors = [COLORS["wavelength"][450], COLORS["wavelength"][532],
                  COLORS["wavelength"][650]]
        patch.set_facecolor(colors[wl_idx])
        patch.set_alpha(0.6)

    ax.set_ylabel("log₁₀(Voltage)")
    ax.set_title("Channel Signal Distributions")
    ax.grid(True, axis="y")

    # ── Panel 2: Channel correlation matrix ──────────────────────────
    ax = axes[1]
    corr = np.corrcoef(raw_channels.T)
    short_names = [n.replace("V_", "").replace("nm_", "\n")
                   for n in channel_names]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(short_names, fontsize=7)

    # Annotate values
    for i in range(9):
        for j in range(9):
            color = "white" if abs(corr[i, j]) > 0.7 else "black"
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Channel Correlation Matrix")

    # ── Panel 3: Mean signal per channel (3λ × 3θ grid) ─────────────
    ax = axes[2]
    mean_signal = raw_channels.mean(axis=0).reshape(3, 3)
    wl_labels = ["450 nm", "532 nm", "650 nm"]
    ang_labels = ["30°", "60°", "90°"]

    im2 = ax.imshow(mean_signal, cmap="YlOrRd", aspect="auto",
                    norm=LogNorm(vmin=max(mean_signal.min(), 1e-15),
                                 vmax=mean_signal.max()))
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(ang_labels)
    ax.set_yticklabels(wl_labels)

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mean_signal[i,j]:.2e}", ha="center",
                    va="center", fontsize=8, color="black")

    fig.colorbar(im2, ax=ax, shrink=0.8, label="Mean Voltage")
    ax.set_xlabel("Detector Angle")
    ax.set_ylabel("Laser Wavelength")
    ax.set_title("Mean Scattering Signal (3λ × 3θ)")

    fig.suptitle("Sensor Signal Analysis — 9-Channel Scattering Matrix",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, output_path)
