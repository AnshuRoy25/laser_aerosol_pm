"""
Plot: ML Model Results
=======================
Prediction scatter plots, feature importance, ablation study bars,
humidity compensation analysis, and model comparison dashboard.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score

from visualization.style import COLORS, save_figure


def plot_ml_results(
    results: dict,
    y_test: np.ndarray,
    y_pred: dict,
    output_path: str = "results/ml_results.png"
):
    """
    Scatter plots of true vs predicted PM2.5 for each model.

    Parameters
    ----------
    results : dict
        From MiniLidarMLPipeline.train_all()
    y_test : array
        True PM2.5 values.
    y_pred : dict
        {model_name: predicted_array}
    """
    n_models = len(y_pred)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    model_colors = COLORS["models"]
    fallback = COLORS["primary"]

    for idx, (ax, (name, pred)) in enumerate(zip(axes, y_pred.items())):
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        c = model_colors.get(name, fallback[idx % len(fallback)])

        ax.scatter(y_test, pred, alpha=0.35, s=15, color=c,
                   edgecolors="none")

        # Perfect prediction line
        lims = [min(y_test.min(), pred.min()) * 0.9,
                max(y_test.max(), pred.max()) * 1.1]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.4)

        # ±20% bounds
        x_line = np.linspace(lims[0], lims[1], 100)
        ax.fill_between(x_line, x_line * 0.8, x_line * 1.2,
                        alpha=0.06, color="green")

        ax.set_xlabel("True PM2.5 (µg/m³)")
        ax.set_ylabel("Predicted PM2.5 (µg/m³)")
        ax.set_title(f"{name}\nMAE = {mae:.1f} µg/m³   R² = {r2:.3f}")
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    fig.suptitle("PM2.5 Prediction — Model Comparison",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_feature_importance(
    importances: dict,
    top_n: int = 15,
    output_path: str = "results/feature_importance.png"
):
    """
    Horizontal bar chart of top feature importances with
    colour-coded feature categories.
    """
    names = list(importances.keys())[:top_n]
    values = list(importances.values())[:top_n]

    # Colour by category
    cat_colors = {
        "V_":   "#2E86AB",   # raw channels
        "SR_":  "#A23B72",   # spectral ratios
        "AR_":  "#F18F01",   # angular ratios
        "AE_":  "#3B9979",   # Ångström
        "total": "#C73E1D",  # aggregates
        "fine":  "#C73E1D",
        "fwd":   "#C73E1D",
        "temp":  "#666666",  # environmental
        "rh":    "#666666",
        "dew":   "#666666",
        "abs":   "#666666",
    }

    bar_colors = []
    for n in names:
        matched = False
        for prefix, c in cat_colors.items():
            if n.lower().startswith(prefix.lower()):
                bar_colors.append(c)
                matched = True
                break
        if not matched:
            bar_colors.append("#999999")

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=bar_colors, alpha=0.85, height=0.7,
            edgecolor="white", linewidth=0.5)

    # Value labels
    for i, (v, n) in enumerate(zip(values, names)):
        ax.text(v + max(values) * 0.01, i, f"{v:.3f}",
                va="center", fontsize=8, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest PM2.5")
    ax.grid(True, axis="x")

    # Legend for categories
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#2E86AB", label="Raw channels"),
        Patch(facecolor="#A23B72", label="Spectral ratios"),
        Patch(facecolor="#F18F01", label="Angular ratios"),
        Patch(facecolor="#3B9979", label="Ångström exponents"),
        Patch(facecolor="#C73E1D", label="Aggregate features"),
        Patch(facecolor="#666666", label="Environmental"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8)

    fig.tight_layout()
    save_figure(fig, output_path)


def plot_ablation_study(
    ablation_results: dict,
    output_path: str = "results/ablation_study.png"
):
    """
    Bar chart showing MAE degradation when each feature group is removed.

    Parameters
    ----------
    ablation_results : dict
        From MiniLidarMLPipeline.ablation_study()
    """
    if "full" not in ablation_results:
        print("    [SKIP] No ablation results to plot.")
        return

    baseline_mae = ablation_results["full"]["mae"]

    groups = []
    deltas = []
    mae_vals = []
    for key, data in ablation_results.items():
        if key == "full":
            continue
        label = key.replace("remove_", "").replace("_", " ").title()
        groups.append(label)
        deltas.append(data.get("delta_pct", 0))
        mae_vals.append(data["mae"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    y_pos = np.arange(len(groups))
    colors = ["#C73E1D" if d > 0 else "#3B9979" for d in deltas]

    # ── Panel 1: Delta MAE (%) ───────────────────────────────────────
    ax1.barh(y_pos, deltas, color=colors, alpha=0.8, height=0.6)
    ax1.axvline(x=0, color="#333333", linewidth=0.8)

    for i, (d, m) in enumerate(zip(deltas, mae_vals)):
        ax1.text(d + (1 if d >= 0 else -1), i,
                 f"{d:+.1f}%", va="center", fontsize=9,
                 color=colors[i], fontweight="bold")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(groups, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Change in MAE (%)")
    ax1.set_title("MAE Degradation When Feature Group Removed")
    ax1.grid(True, axis="x")

    # ── Panel 2: Absolute MAE comparison ─────────────────────────────
    all_names = ["Full Model"] + groups
    all_maes = [baseline_mae] + mae_vals
    bar_colors = ["#2E86AB"] + ["#F18F01"] * len(groups)

    ax2.barh(np.arange(len(all_names)), all_maes, color=bar_colors,
             alpha=0.8, height=0.6)

    for i, m in enumerate(all_maes):
        ax2.text(m + max(all_maes) * 0.01, i, f"{m:.1f}",
                 va="center", fontsize=9)

    ax2.set_yticks(np.arange(len(all_names)))
    ax2.set_yticklabels(all_names, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("MAE (µg/m³)")
    ax2.set_title("Absolute MAE — Ablation Study")
    ax2.grid(True, axis="x")

    fig.suptitle("Feature Group Ablation Study — PM2.5 Prediction",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_humidity_analysis(
    humidity_results: dict,
    output_path: str = "results/humidity_analysis.png"
):
    """
    Grouped bar chart comparing baseline vs ML-corrected MAE
    across RH bins, plus improvement factor line.

    Parameters
    ----------
    humidity_results : dict
        From MiniLidarMLPipeline.humidity_analysis()
    """
    if not humidity_results:
        print("    [SKIP] No humidity results to plot.")
        return

    labels = list(humidity_results.keys())
    mae_baseline = [v["mae_baseline"] for v in humidity_results.values()]
    mae_ml = [v["mae_ml"] for v in humidity_results.values()]
    improvement = [v["improvement_factor"] for v in humidity_results.values()]
    counts = [v["n"] for v in humidity_results.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel 1: MAE comparison bars ─────────────────────────────────
    bars1 = ax1.bar(x - width / 2, mae_baseline, width, label="Baseline (single-channel)",
                    color="#C73E1D", alpha=0.8)
    bars2 = ax1.bar(x + width / 2, mae_ml, width, label="ML-corrected (full model)",
                    color="#3B9979", alpha=0.8)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1,
                 f"{h:.0f}", ha="center", fontsize=8, color="#C73E1D")
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1,
                 f"{h:.0f}", ha="center", fontsize=8, color="#3B9979")

    # Sample counts
    for i, n in enumerate(counts):
        ax1.text(i, -max(mae_baseline) * 0.05, f"n={n}", ha="center",
                 fontsize=8, color="#666666")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Relative Humidity Range")
    ax1.set_ylabel("MAE (µg/m³)")
    ax1.set_title("PM2.5 MAE: Baseline vs ML-Corrected")
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y")

    # ── Panel 2: Improvement factor ──────────────────────────────────
    bar_colors = ["#2E86AB" if imp >= 1 else "#C73E1D" for imp in improvement]
    ax2.bar(x, improvement, color=bar_colors, alpha=0.8, width=0.5)
    ax2.axhline(y=1.0, color="#333333", linewidth=0.8, linestyle="--")

    for i, imp in enumerate(improvement):
        ax2.text(i, imp + 0.05, f"{imp:.1f}×", ha="center", fontsize=10,
                 fontweight="bold", color=bar_colors[i])

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Relative Humidity Range")
    ax2.set_ylabel("Improvement Factor (Baseline MAE / ML MAE)")
    ax2.set_title("Humidity Compensation Effectiveness")
    ax2.grid(True, axis="y")

    fig.suptitle("Humidity Compensation Analysis — Mini-LIDAR ML Pipeline",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_model_comparison_dashboard(
    results: dict,
    output_path: str = "results/model_dashboard.png"
):
    """
    Summary dashboard: bar chart of MAE/R² for all models × both targets.

    Parameters
    ----------
    results : dict
        From MiniLidarMLPipeline.train_all()
    """
    model_names = list(results.keys())
    n = len(model_names)
    model_colors = COLORS["models"]
    fallback = COLORS["primary"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(n)
    width = 0.35

    for ax, target, title in [(axes[0], "pm25", "PM2.5"),
                               (axes[1], "pm10", "PM10")]:
        maes = [results[m][target]["mae"] for m in model_names]
        r2s = [results[m][target]["r2"] for m in model_names]
        colors = [model_colors.get(m, fallback[i % len(fallback)])
                  for i, m in enumerate(model_names)]

        # MAE bars
        bars = ax.bar(x, maes, width=0.6, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=0.5)

        for bar, mae_val, r2_val in zip(bars, maes, r2s):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(maes) * 0.02,
                    f"MAE={mae_val:.1f}\nR²={r2_val:.3f}",
                    ha="center", fontsize=8, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=9)
        ax.set_ylabel("MAE (µg/m³)")
        ax.set_title(f"{title} Prediction")
        ax.grid(True, axis="y")

    fig.suptitle("Model Performance Comparison Dashboard",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, output_path)
