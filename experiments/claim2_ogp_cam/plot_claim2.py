"""
plot_claim2.py
==============
Generate publication-quality figures for Claim 2:
  "OGP + CAM truly protect old-knowledge subspace"

Panels:
  Fig 1 (E1): Energy Retention vs Task-B step, 4 conditions × 4 levels
  Fig 2 (E2): Principal Angle vs Task-B step, 4 conditions × 4 levels
  Fig 3 (E3): Subspace Overlap vs Task-B step, 4 conditions × 4 levels
  Fig 4 (E4): Param Drift vs Task-B step, 4 conditions × 4 levels
  Fig 5 (main): 2x2 summary: cms_fast + cms_slow, E1 + E4, all conditions
  Fig 6 (trade-off): Retention Rate vs Task-B final loss (plasticity) scatter

Usage:
  python plot_claim2.py --results_dir ~/experiments/results_claim2 \
                        --output_dir  ~/experiments/figs_claim2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Style ──────────────────────────────────────────────────────────────────────
try:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
except Exception:
    pass

CONDITION_COLORS = {
    "vanilla":  "#E74C3C",   # red
    "ogp_only": "#3498DB",   # blue
    "cam_only": "#2ECC71",   # green
    "full_hgc": "#8E44AD",   # purple
}
CONDITION_LABELS = {
    "vanilla":  "Vanilla",
    "ogp_only": "OGP only",
    "cam_only": "CAM only",
    "full_hgc": "Full HGC",
}
CONDITION_LINESTYLES = {
    "vanilla":  "-",
    "ogp_only": "--",
    "cam_only": "-.",
    "full_hgc": ":",
}
CONDITIONS = ["vanilla", "ogp_only", "cam_only", "full_hgc"]
LEVEL_NAMES   = ["cms_fast", "cms_mid", "cms_slow", "cms_ultra"]
LEVEL_LABELS  = ["C=1 (fast)", "C=4 (mid)", "C=32 (slow)", "C=128 (ultra)"]
LEVEL_COLORS  = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_records(results_dir: Path) -> Dict[str, Optional[dict]]:
    records = {}
    for cond in CONDITIONS:
        p = results_dir / f"records_{cond}.json"
        if p.exists():
            with open(p) as f:
                records[cond] = json.load(f)
            print(f"  Loaded {cond}: {len(records[cond].get('step', []))} steps")
        else:
            records[cond] = None
            print(f"  Missing: {cond}")
    return records


def ema(values: list, alpha: float = 0.2) -> list:
    """Exponential moving average smoothing."""
    out = []
    v = None
    for x in values:
        if x is None:
            out.append(None)
            continue
        v = x if v is None else alpha * x + (1 - alpha) * v
        out.append(v)
    return out


def extract_series(rec: dict, key: str, smooth: bool = False) -> tuple[list, list]:
    steps = rec.get("step", [])
    vals  = rec.get(key, [])
    xs, ys = [], []
    for s, v in zip(steps, vals):
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            xs.append(s)
            ys.append(v)
    if smooth and ys:
        ys = ema(ys)
    return xs, ys


# ── Plot helpers ───────────────────────────────────────────────────────────────

def plot_metric_grid(records, metric_prefix, ylabel, title, output_path,
                     smooth=True, ylim=None):
    """4×4 grid: rows=conditions, cols=levels. Or 2×4: rows=levels, cols=conditions."""
    # Layout: 2 rows (cms_fast, cms_slow), 4 cols (conditions) for compact figure
    levels_to_show = ["cms_fast", "cms_mid", "cms_slow", "cms_ultra"]
    n_levels = len(levels_to_show)
    n_cond   = len(CONDITIONS)

    fig, axes = plt.subplots(n_cond, n_levels, figsize=(14, 9), sharex=True)

    for ci, cond in enumerate(CONDITIONS):
        rec = records.get(cond)
        for li, (ln, ll) in enumerate(zip(levels_to_show, LEVEL_LABELS)):
            ax = axes[ci][li]
            if rec is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                continue
            xs, ys = extract_series(rec, f"{metric_prefix}_{ln}", smooth=smooth)
            if xs:
                ax.plot(xs, ys, color=CONDITION_COLORS[cond], lw=1.5)
            if metric_prefix == "energy_retention":
                theory = rec.get(f"theory_decay_{ln}", [])
                t_xs   = rec.get("step", [])
                if theory and t_xs:
                    t_pairs = [(s, v) for s, v in zip(t_xs, theory) if v is not None]
                    ax.plot([p[0] for p in t_pairs], [p[1] for p in t_pairs],
                            color="gray", lw=0.8, ls=":", alpha=0.7, label="β^k theory")
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_title(f"{CONDITION_LABELS[cond]}\n{ll}", fontsize=9)
            if ci == n_cond - 1:
                ax.set_xlabel("Task-B step")
            if li == 0:
                ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, lw=0.5)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_condition_overlay(records, metric_prefix, ylabel, title, output_path,
                            smooth=True, ylim=None, levels=None):
    """One subplot per level; all conditions overlaid."""
    levels_plot  = levels or LEVEL_NAMES
    level_labels = [LEVEL_LABELS[LEVEL_NAMES.index(ln)] for ln in levels_plot]
    ncols = len(levels_plot)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, (ln, ll) in zip(axes, zip(levels_plot, level_labels)):
        for cond in CONDITIONS:
            rec = records.get(cond)
            if rec is None:
                continue
            xs, ys = extract_series(rec, f"{metric_prefix}_{ln}", smooth=smooth)
            if xs:
                ax.plot(xs, ys,
                        color=CONDITION_COLORS[cond],
                        ls=CONDITION_LINESTYLES[cond],
                        lw=1.8, label=CONDITION_LABELS[cond])
        # Theory decay (vanilla as reference)
        if metric_prefix == "energy_retention":
            rec0 = records.get("vanilla") or next(
                (r for r in records.values() if r is not None), None)
            if rec0:
                theory = rec0.get(f"theory_decay_{ln}", [])
                t_xs   = rec0.get("step", [])
                if theory and t_xs:
                    ax.plot(t_xs, theory, color="black", lw=0.8, ls=":", alpha=0.5,
                            label="β^k theory")
        ax.set_title(ll)
        ax.set_xlabel("Task-B step")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.legend(fontsize=9, loc="upper right")

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_main_summary(records, output_path):
    """2×2 main figure: E1 energy retention + E4 param drift, fast + slow levels."""
    pairs = [
        ("energy_retention", "cms_fast", "C=1 (fast)", "Energy Retention\n||U_A^T m_t||² / ||U_A^T m_0||²"),
        ("energy_retention", "cms_slow", "C=32 (slow)", "Energy Retention\n||U_A^T m_t||² / ||U_A^T m_0||²"),
        ("param_drift",      "cms_fast", "C=1 (fast)", "Param Drift\n||θ_t - θ_A|| / ||θ_A||"),
        ("param_drift",      "cms_slow", "C=32 (slow)", "Param Drift\n||θ_t - θ_A|| / ||θ_A||"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes_flat = axes.flatten()

    for ax, (metric, ln, ll, ylbl) in zip(axes_flat, pairs):
        for cond in CONDITIONS:
            rec = records.get(cond)
            if rec is None:
                continue
            xs, ys = extract_series(rec, f"{metric}_{ln}", smooth=True)
            if xs:
                ax.plot(xs, ys,
                        color=CONDITION_COLORS[cond],
                        ls=CONDITION_LINESTYLES[cond],
                        lw=2.0, label=CONDITION_LABELS[cond])
        if metric == "energy_retention":
            rec0 = records.get("vanilla") or next(
                (r for r in records.values() if r is not None), None)
            if rec0:
                theory = rec0.get(f"theory_decay_{ln}", [])
                t_xs   = rec0.get("step", [])
                if theory and t_xs:
                    ax.plot(t_xs, theory, color="black", lw=0.9, ls=":", alpha=0.6,
                            label="β^k (theory)")
        ax.set_title(f"{metric.replace('_', ' ').title()} — {ll}", fontsize=11)
        ax.set_xlabel("Task-B step", fontsize=10)
        ax.set_ylabel(ylbl, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3, lw=0.5)

    fig.suptitle(
        "Claim 2: OGP + CAM protect old-knowledge subspace (vocab-split, 40M)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_tradeoff(records, output_path):
    """Plasticity-retention trade-off scatter: x=final loss_B, y=final energy_retention."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for level_idx, (ln, ll) in enumerate([("cms_fast", "C=1 (fast)"),
                                          ("cms_slow", "C=32 (slow)")]):
        ax = axes[level_idx]
        for cond in CONDITIONS:
            rec = records.get(cond)
            if rec is None:
                continue
            loss_b = [v for v in rec.get("loss_B", []) if v is not None]
            er_vals = [v for v in rec.get(f"energy_retention_{ln}", []) if v is not None]
            if not loss_b or not er_vals:
                continue
            # Use last-20-step average
            final_loss = float(np.mean(loss_b[-20:]))
            final_er   = float(np.mean(er_vals[-20:]))
            ax.scatter([final_loss], [final_er],
                       color=CONDITION_COLORS[cond], s=120, zorder=5,
                       label=CONDITION_LABELS[cond])
            ax.annotate(CONDITION_LABELS[cond], (final_loss, final_er),
                        textcoords="offset points", xytext=(6, 3), fontsize=9)

        ax.set_xlabel("Final Task-B loss (plasticity proxy)")
        ax.set_ylabel("Energy Retention (final avg)")
        ax.set_title(f"Plasticity–Retention Trade-off ({ll})")
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.legend(fontsize=9)

    fig.suptitle("Claim 2: CAM improves retention without sacrificing plasticity",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_loss_curves(records, output_path):
    """Task A (forgetting) and Task B (plasticity) loss curves for all conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for cond in CONDITIONS:
        rec = records.get(cond)
        if rec is None:
            continue
        # Task B loss
        xs_b = rec.get("step", [])
        ys_b = [v for v in rec.get("loss_B", []) if v is not None]
        if xs_b and ys_b:
            ys_b_s = ema(ys_b, alpha=0.15)
            axes[0].plot(xs_b[:len(ys_b_s)], ys_b_s,
                         color=CONDITION_COLORS[cond],
                         ls=CONDITION_LINESTYLES[cond],
                         lw=1.8, label=CONDITION_LABELS[cond])
        # Task A forgetting
        xs_a = rec.get("loss_A_step", [])
        ys_a = rec.get("loss_A_forgetting", [])
        if xs_a and ys_a:
            axes[1].plot(xs_a, ys_a,
                         color=CONDITION_COLORS[cond],
                         ls=CONDITION_LINESTYLES[cond],
                         lw=1.8, label=CONDITION_LABELS[cond], marker="o", ms=4)

    axes[0].set_title("Task B Loss (Plasticity)", fontsize=12)
    axes[0].set_xlabel("Task-B step")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, lw=0.5)

    axes[1].set_title("Task A Loss During Task B (Forgetting)", fontsize=12)
    axes[1].set_xlabel("Task-B step")
    axes[1].set_ylabel("Cross-entropy loss on Task A")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, lw=0.5)

    fig.suptitle("Claim 2: Plasticity & Forgetting under HGC conditions",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results_claim2")
    parser.add_argument("--output_dir",  default="./figs_claim2")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    output_dir  = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading records ===")
    records = load_records(results_dir)
    available = [c for c in CONDITIONS if records[c] is not None]
    print(f"  Available: {available}")

    if not available:
        print("No records found. Exiting.")
        return

    print("\n=== Generating figures ===")

    # E1: Energy Retention — condition overlay per level
    plot_condition_overlay(
        records, "energy_retention",
        ylabel="||U_A^T m_t||² / ||U_A^T m_0||²",
        title="E1: Subspace Energy Retention (all conditions)",
        output_path=output_dir / "claim2_E1_energy_retention.png",
        smooth=True, ylim=(0, None),
    )

    # E2: Principal Angle
    plot_condition_overlay(
        records, "principal_angle",
        ylabel="Min Principal Angle (degrees)",
        title="E2: Principal Angle to Old-Task Subspace",
        output_path=output_dir / "claim2_E2_principal_angle.png",
        smooth=True,
    )

    # E3: Subspace Overlap
    plot_condition_overlay(
        records, "subspace_overlap",
        ylabel="||U_A^T U_t||_F / √r",
        title="E3: Structural Subspace Overlap",
        output_path=output_dir / "claim2_E3_subspace_overlap.png",
        smooth=True, ylim=(0, None),
    )

    # E4: Param Drift
    plot_condition_overlay(
        records, "param_drift",
        ylabel="||θ_t - θ_A|| / ||θ_A||",
        title="E4: Parameter Drift (forgetting proxy)",
        output_path=output_dir / "claim2_E4_param_drift.png",
        smooth=True, ylim=(0, None),
    )

    # Main summary figure
    plot_main_summary(
        records,
        output_path=output_dir / "claim2_main_summary.png",
    )

    # Loss curves
    plot_loss_curves(
        records,
        output_path=output_dir / "claim2_loss_curves.png",
    )

    # Trade-off scatter
    plot_tradeoff(
        records,
        output_path=output_dir / "claim2_tradeoff.png",
    )

    # Fast levels only (cleaner for paper)
    plot_condition_overlay(
        records, "energy_retention",
        ylabel="Energy Retention",
        title="E1: Energy Retention (cms_fast and cms_slow)",
        output_path=output_dir / "claim2_E1_fast_slow.png",
        smooth=True, ylim=(0, None),
        levels=["cms_fast", "cms_slow"],
    )

    print(f"\n=== Done. Figures saved to {output_dir} ===")


if __name__ == "__main__":
    main()
