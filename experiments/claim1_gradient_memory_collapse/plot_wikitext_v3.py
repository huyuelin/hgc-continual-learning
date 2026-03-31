"""
plot_wikitext_v3.py
===================
Publication-quality figures for the WikiText-2 three-phase experiments.
Reads results_phase{1,2,3}/records.json and produces:

  wt2_E1_momentum_decay.{pdf,png}      — E1: momentum subspace projection decay
  wt2_E2_collapse_vs_forgetting.{pdf,png} — E2: collapse precedes forgetting
  wt2_E3_hierarchy.{pdf,png}           — E3: all CMS levels same β decay rate
  wt2_main_figure.{pdf,png}            — combined 4-panel main figure (Phase 3)
  wt2_appendix_phase1.{pdf,png}        — appendix figure (Phase 1)

Usage:
    python plot_wikitext_v3.py --results_dir ~/experiments --output_dir ./figs_wikitext
    python plot_wikitext_v3.py --results_dir ~/experiments --output_dir ./figs_wikitext --phase 1
"""
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─── Typography ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "legend.fontsize":    8.5,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "axes.linewidth":     0.8,
    "lines.linewidth":    1.6,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.35,
    "figure.dpi":         150,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# ─── Constants ────────────────────────────────────────────────────────────────
LEVEL_NAMES   = ["cms_fast", "cms_mid", "cms_slow", "cms_ultra"]
LEVEL_PERIODS = [1, 4, 32, 128]
LEVEL_LABELS  = ["C=1 (fast)", "C=4 (mid)", "C=32 (slow)", "C=128 (ultra)"]
LEVEL_COLORS  = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
BETA          = 0.9

PHASE_TAGS = {1: "~48M", 2: "~256M", 3: "~742M"}
PHASE_DIMS = {1: "dim=256, 8-layer", 2: "dim=512, 12-layer", 3: "dim=768, 16-layer"}


# ─── Utilities ────────────────────────────────────────────────────────────────

def ema(x: List[float], alpha: float = 0.15) -> List[float]:
    out, s = [], float("nan")
    for v in x:
        if v is None or math.isnan(v):
            out.append(float("nan"))
            continue
        s = v if math.isnan(s) else alpha * v + (1 - alpha) * s
        out.append(s)
    return out


def safe(lst: List, default=float("nan")) -> np.ndarray:
    """Convert list with None/nan to numpy array."""
    return np.array([v if v is not None else float("nan") for v in lst], dtype=float)


def load_records(results_dir: str, phase: int) -> Optional[Dict]:
    path = Path(results_dir) / f"results_phase{phase}" / "records.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def theory_curve(steps: np.ndarray, C: int, beta: float = BETA) -> np.ndarray:
    k_eff = steps // C + 1
    return beta ** k_eff


# ─── Figure helpers ───────────────────────────────────────────────────────────

def add_panel_label(ax, label: str, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="right")


def style_ax(ax, xlabel="", ylabel="", title="", ylim=None, yticks=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(True, axis="y")
    ax.grid(True, axis="x", alpha=0.2)


# ─── E1: Momentum subspace projection decay ──────────────────────────────────

def plot_E1(rec: Dict, output_dir: Path, suffix: str = ""):
    steps  = np.array(rec["step"])
    tag    = rec.get("tag", "")
    phase  = rec.get("phase", 1)
    n_p    = rec.get("n_params_M", 0)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    fig.suptitle(
        f"E1 · Momentum Subspace Projection Decay  [{PHASE_TAGS[phase]}, {PHASE_DIMS[phase]}]",
        fontweight="bold", y=1.02
    )

    # Left: wall-clock step decay (log scale)
    ax = axes[0]
    active_levels = []
    for ln, C, label, color in zip(LEVEL_NAMES, LEVEL_PERIODS, LEVEL_LABELS, LEVEL_COLORS):
        vals = safe(rec.get(f"mom_proj_norm_{ln}", []))
        if np.all(np.isnan(vals)):
            continue
        active_levels.append((ln, C, label, color))
        # EMA smoothed
        sm = np.array(ema(vals.tolist(), alpha=0.2))
        ax.semilogy(steps, sm, color=color, label=label, alpha=0.85)
        # Theory
        th = theory_curve(steps, C)
        ax.semilogy(steps, th, color=color, linestyle="--", linewidth=0.9, alpha=0.5)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, label="Initial (=1)")
    add_panel_label(ax, "(a)")
    style_ax(ax,
             xlabel="Task B step",
             ylabel=r"$\|U_A^T m_t\| / \|m_0\|$",
             title="Wall-clock decay (log)")
    ax.legend(loc="lower left", ncol=1, handlelength=1.5)
    # Annotate β
    ax.text(0.65, 0.92, r"$\beta^{k_\mathrm{eff}}$ theory (dashed)",
            transform=ax.transAxes, fontsize=7.5, color="gray")

    # Right: per-effective-update decay
    ax = axes[1]
    for ln, C, label, color in active_levels:
        vals = safe(rec.get(f"mom_proj_norm_{ln}", []))
        k_eff = steps // C + 1
        mask = ~np.isnan(vals)
        if mask.sum() < 2:
            continue
        sm = np.array(ema(vals.tolist(), alpha=0.2))
        ax.semilogy(k_eff, sm, color=color, label=label, alpha=0.85)
        # Theory
        ax.semilogy(np.arange(1, k_eff[-1] + 1),
                    BETA ** np.arange(1, k_eff[-1] + 1),
                    color=color, linestyle="--", linewidth=0.9, alpha=0.5)

    add_panel_label(ax, "(b)")
    style_ax(ax,
             xlabel=r"Effective updates $k_\mathrm{eff} = \lfloor t/C \rfloor + 1$",
             ylabel=r"$\|U_A^T m_t\| / \|m_0\|$",
             title=r"Per-effective-update (log), $\beta=0.9$")
    ax.legend(loc="lower left", ncol=1, handlelength=1.5)

    fig.tight_layout()
    stem = f"wt2_E1_momentum_decay{suffix}"
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)
    print(f"  Saved {stem}.pdf/png")


# ─── E2: Momentum collapse precedes parameter forgetting ─────────────────────

def plot_E2(rec: Dict, output_dir: Path, suffix: str = ""):
    steps = np.array(rec["step"])
    phase = rec.get("phase", 1)

    # Determine active levels
    active_levels = []
    for ln, C, label, color in zip(LEVEL_NAMES, LEVEL_PERIODS, LEVEL_LABELS, LEVEL_COLORS):
        vals = safe(rec.get(f"mom_proj_norm_{ln}", []))
        if np.all(np.isnan(vals)):
            continue
        active_levels.append((ln, C, label, color))

    n_cols = len(active_levels)
    if n_cols == 0:
        print("  E2: no active levels, skipping")
        return

    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.2))
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(
        f"E2 · Optimizer-State Collapse Precedes Parameter Forgetting  [{PHASE_TAGS[phase]}]",
        fontweight="bold", y=1.03
    )

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for i, (ln, C, label, color) in enumerate(active_levels):
        ax = axes[i]
        ax2 = ax.twinx()

        mom_vals  = safe(rec.get(f"mom_proj_norm_{ln}", []))
        drift_vals = safe(rec.get(f"param_drift_{ln}", []))

        mom_sm   = np.array(ema(mom_vals.tolist(), alpha=0.15))
        drift_sm = np.array(ema(drift_vals.tolist(), alpha=0.15))

        line1, = ax.plot(steps, mom_sm, color=color, linewidth=1.8,
                         label=r"$\|U_A^T m_t\| / \|m_0\|$")
        line2, = ax2.plot(steps, drift_sm, color="dimgray", linewidth=1.4,
                          linestyle="--", label=r"$\|\theta_t - \theta_A\| / \|\theta_A\|$")

        ax.set_xlabel("Task B step")
        ax.set_ylabel(r"Momentum proj. (normalized)", color=color)
        ax2.set_ylabel(r"Param. drift", color="dimgray")
        ax.tick_params(axis="y", labelcolor=color)
        ax2.tick_params(axis="y", labelcolor="dimgray")

        ax.set_title(f"{label}", fontsize=10)
        ax.grid(True, axis="x", alpha=0.2)
        ax.grid(True, axis="y", alpha=0.3)
        ax2.spines["top"].set_visible(False)
        ax.spines["top"].set_visible(False)

        lines = [line1, line2]
        labs  = [l.get_label() for l in lines]
        ax.legend(lines, labs, loc="center right", fontsize=7.5, handlelength=1.2)
        add_panel_label(ax, panel_labels[i])

    fig.tight_layout()
    stem = f"wt2_E2_collapse_vs_forgetting{suffix}"
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)
    print(f"  Saved {stem}.pdf/png")


# ─── E3: All CMS levels same β decay rate ────────────────────────────────────

def plot_E3(rec: Dict, output_dir: Path, suffix: str = ""):
    steps = np.array(rec["step"])
    phase = rec.get("phase", 1)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    fig.suptitle(
        f"E3 · All CMS Levels Share Identical Per-Update Decay Rate (≈β)  [{PHASE_TAGS[phase]}]",
        fontweight="bold", y=1.02
    )

    # Left: apparent per-step persistence (different C → different apparent half-life)
    ax = axes[0]
    active_levels = []
    for ln, C, label, color in zip(LEVEL_NAMES, LEVEL_PERIODS, LEVEL_LABELS, LEVEL_COLORS):
        vals = safe(rec.get(f"mom_proj_norm_{ln}", []))
        if np.all(np.isnan(vals)):
            continue
        active_levels.append((ln, C, label, color, vals))
        sm = np.array(ema(vals.tolist(), alpha=0.15))
        ax.plot(steps, sm, color=color, label=label)

    add_panel_label(ax, "(a)")
    style_ax(ax,
             xlabel="Task B step",
             ylabel=r"$\|U_A^T m_t\| / \|m_0\|$",
             title="Apparent decay (wall-clock)")
    ax.legend(loc="upper right", ncol=1, handlelength=1.5)
    ax.text(0.35, 0.50,
            "Different step-scales → different\napparent half-lives",
            transform=ax.transAxes, fontsize=7.5, color="gray",
            va="center", ha="left")

    # Right: scatter of per-effective-update decay vs β
    ax = axes[1]
    beta_estimates = []
    for ln, C, label, color, vals in active_levels:
        # Fit: log(proj_norm) = k_eff * log(β_hat) → slope = log(β_hat)
        k_eff = (steps // C + 1).astype(float)
        log_vals = np.log(np.maximum(vals, 1e-8))
        mask = np.isfinite(log_vals) & (log_vals > -10)
        if mask.sum() < 10:
            continue
        slope, _ = np.polyfit(k_eff[mask], log_vals[mask], 1)
        beta_hat = math.exp(slope)
        beta_estimates.append((C, beta_hat, label, color))

    if beta_estimates:
        xs = [e[0] for e in beta_estimates]
        ys = [e[1] for e in beta_estimates]
        colors_s = [e[3] for e in beta_estimates]
        labels_s = [e[2] for e in beta_estimates]

        for x, y, c, lbl in zip(xs, ys, colors_s, labels_s):
            ax.scatter(x, y, color=c, s=80, zorder=5, label=lbl)
            ax.annotate(f"β̂={y:.3f}", xy=(x, y), xytext=(5, 5),
                        textcoords="offset points", fontsize=7.5, color=c)

        ax.axhline(BETA, color="black", linestyle="--", linewidth=1.2,
                   label=f"β={BETA} (DeepMomentum)")
        ax.set_xscale("log")
        ax.set_xlabel("Update period C")
        ax.set_ylabel(r"Fitted $\hat{\beta}$ per effective update")
        ax.set_title(r"Per-effective-update $\hat{\beta}$ ≈ 0.9 for all levels")
        ax.legend(loc="lower right", ncol=1, handlelength=1.5)
        ax.set_ylim(0.7, 1.0)
        ax.grid(True, axis="y")
        ax.grid(True, axis="x", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        add_panel_label(ax, "(b)")

        # Print fitted β values
        print(f"  E3 fitted β values:")
        for x, y, c, lbl in zip(xs, ys, colors_s, labels_s):
            print(f"    {lbl}: β̂={y:.4f}")
    else:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="gray")

    fig.tight_layout()
    stem = f"wt2_E3_hierarchy{suffix}"
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)
    print(f"  Saved {stem}.pdf/png")


# ─── Main 4-panel figure (best phase) ────────────────────────────────────────

def plot_main_figure(rec: Dict, output_dir: Path, suffix: str = ""):
    steps  = np.array(rec["step"])
    phase  = rec.get("phase", 3)
    n_p    = rec.get("n_params_M", 0)
    tag    = rec.get("tag", "")

    # Determine active levels
    active_levels = []
    for ln, C, label, color in zip(LEVEL_NAMES, LEVEL_PERIODS, LEVEL_LABELS, LEVEL_COLORS):
        vals = safe(rec.get(f"mom_proj_norm_{ln}", []))
        if not np.all(np.isnan(vals)):
            active_levels.append((ln, C, label, color))

    # Loss curve
    loss_B = safe(rec.get("loss_B", []))

    fig = plt.figure(figsize=(10, 7))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.38)

    # Panel A: Task B training loss
    ax_loss = fig.add_subplot(gs[0, 0])
    sm_loss = np.array(ema(loss_B.tolist(), alpha=0.1))
    ax_loss.plot(steps, sm_loss, color="#2c7bb6", linewidth=1.6)
    ax_loss.set_xlabel("Task B step")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Task B training loss")
    ax_loss.grid(True, axis="y")
    ax_loss.grid(True, axis="x", alpha=0.2)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)
    add_panel_label(ax_loss, "(A)")

    # Panel B: PPL on Task A validation (forgetting)
    ax_ppl = fig.add_subplot(gs[0, 1])
    ppl_steps = rec.get("ppl_taskA_step", [])
    ppl_vals  = rec.get("ppl_taskA_eval", [])
    if ppl_steps and ppl_vals:
        ppl_arr = np.array([v if v < 1e6 else np.nan for v in ppl_vals])
        ax_ppl.plot(ppl_steps, ppl_arr, color="#d7191c", linewidth=1.8, marker="o",
                    markersize=4, label="PPL on Task A valid.")
        ppl_A_baseline = rec.get("ppl_after_taskA", None)
        if ppl_A_baseline and ppl_A_baseline < 1e6:
            ax_ppl.axhline(ppl_A_baseline, color="gray", linestyle="--", linewidth=1.0,
                           label=f"After Task A ({ppl_A_baseline:.1f})")
        ax_ppl.set_xlabel("Task B step")
        ax_ppl.set_ylabel("Perplexity")
        ax_ppl.set_title("Forgetting (PPL on Task A)")
        ax_ppl.legend(loc="upper left", fontsize=7.5)
        ax_ppl.grid(True, axis="y")
        ax_ppl.spines["top"].set_visible(False)
        ax_ppl.spines["right"].set_visible(False)
    else:
        ax_ppl.text(0.5, 0.5, "PPL data\nnot available", transform=ax_ppl.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
    add_panel_label(ax_ppl, "(B)")

    # Panel C: E1 — Momentum decay (log)
    ax_e1 = fig.add_subplot(gs[1, 0])
    for ln, C, label, color in active_levels:
        vals = safe(rec.get(f"mom_proj_norm_{ln}", []))
        sm   = np.array(ema(vals.tolist(), alpha=0.2))
        ax_e1.semilogy(steps, sm, color=color, label=label, linewidth=1.5)
        th   = theory_curve(steps, C)
        ax_e1.semilogy(steps, th, color=color, linestyle="--", linewidth=0.8, alpha=0.5)
    ax_e1.axhline(1.0, color="gray", linestyle=":", linewidth=0.7)
    ax_e1.set_xlabel("Task B step")
    ax_e1.set_ylabel(r"$\|U_A^T m_t\| / \|m_0\|$")
    ax_e1.set_title(r"E1: Momentum projection decay (log)")
    ax_e1.legend(loc="lower left", ncol=1, fontsize=7.5, handlelength=1.3)
    ax_e1.grid(True, axis="y")
    ax_e1.grid(True, axis="x", alpha=0.2)
    ax_e1.spines["top"].set_visible(False)
    ax_e1.spines["right"].set_visible(False)
    add_panel_label(ax_e1, "(C)")

    # Panel D: E2 — Fast level: collapse vs drift
    ax_e2 = fig.add_subplot(gs[1, 1])
    ax_e2r = ax_e2.twinx()
    ln_fast, C_fast = "cms_fast", 1
    mom_f   = safe(rec.get(f"mom_proj_norm_{ln_fast}", []))
    drift_f = safe(rec.get(f"param_drift_{ln_fast}", []))
    if not np.all(np.isnan(mom_f)):
        sm_m = np.array(ema(mom_f.tolist(), alpha=0.15))
        sm_d = np.array(ema(drift_f.tolist(), alpha=0.15))
        l1, = ax_e2.plot(steps, sm_m, color=LEVEL_COLORS[0], linewidth=1.8,
                         label="Momentum proj. (C=1)")
        l2, = ax_e2r.plot(steps, sm_d, color="dimgray", linewidth=1.4,
                          linestyle="--", label="Param. drift (C=1)")
        ax_e2.set_ylabel(r"Momentum proj.", color=LEVEL_COLORS[0])
        ax_e2r.set_ylabel("Param. drift", color="dimgray")
        ax_e2.tick_params(axis="y", labelcolor=LEVEL_COLORS[0])
        ax_e2r.tick_params(axis="y", labelcolor="dimgray")
        ax_e2.legend([l1, l2], [l1.get_label(), l2.get_label()],
                     loc="center right", fontsize=7.5, handlelength=1.2)
    ax_e2.set_xlabel("Task B step")
    ax_e2.set_title("E2: Collapse precedes forgetting (C=1)")
    ax_e2.grid(True, axis="y")
    ax_e2.grid(True, axis="x", alpha=0.2)
    ax_e2.spines["top"].set_visible(False)
    ax_e2r.spines["top"].set_visible(False)
    add_panel_label(ax_e2, "(D)")

    fig.suptitle(
        f"Gradient Memory Collapse in HOPE  [{PHASE_TAGS[phase]}, {PHASE_DIMS[phase]}]  "
        f"WikiText-2, DeepMomentum β=0.9",
        fontweight="bold", y=1.01, fontsize=12
    )

    stem = f"wt2_main_figure{suffix}"
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)
    print(f"  Saved {stem}.pdf/png")


# ─── Cross-phase comparison ───────────────────────────────────────────────────

def plot_cross_phase(recs: Dict[int, Dict], output_dir: Path):
    """E1 decay comparison across all three phases (for the paper)."""
    if not recs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)
    fig.suptitle(
        "E1 · Momentum Decay Across Model Scales  (WikiText-2, HOPE)",
        fontweight="bold", y=1.03
    )

    phase_labels = ["(a) Phase 1 (~48M)", "(b) Phase 2 (~256M)", "(c) Phase 3 (~742M)"]
    for col, p in enumerate([1, 2, 3]):
        ax = axes[col]
        rec = recs.get(p)
        if rec is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="gray")
            ax.set_title(phase_labels[col])
            continue

        steps = np.array(rec["step"])
        active = False
        for ln, C, label, color in zip(LEVEL_NAMES, LEVEL_PERIODS, LEVEL_LABELS, LEVEL_COLORS):
            vals = safe(rec.get(f"mom_proj_norm_{ln}", []))
            if np.all(np.isnan(vals)):
                continue
            sm = np.array(ema(vals.tolist(), alpha=0.2))
            ax.semilogy(steps, sm, color=color, label=label, linewidth=1.5)
            active = True

        if active:
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.7)
        ax.set_xlabel("Task B step")
        if col == 0:
            ax.set_ylabel(r"$\|U_A^T m_t\| / \|m_0\|$")
        ax.set_title(phase_labels[col])
        ax.grid(True, axis="y")
        ax.grid(True, axis="x", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == 0:
            ax.legend(loc="lower left", fontsize=7.5, handlelength=1.3)

    fig.tight_layout()
    stem = "wt2_cross_phase_E1"
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)
    print(f"  Saved {stem}.pdf/png")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot WikiText-2 Claim 1 results")
    parser.add_argument("--results_dir", default="~/experiments",
                        help="Parent directory containing results_phase{1,2,3}/")
    parser.add_argument("--output_dir",  default="./figs_wikitext",
                        help="Output directory for figures")
    parser.add_argument("--phase",       type=int, default=0,
                        help="Plot only this phase (0=all)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    output_dir  = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    phases_to_plot = [args.phase] if args.phase > 0 else [1, 2, 3]

    all_recs: Dict[int, Dict] = {}
    for p in phases_to_plot:
        rec = load_records(results_dir, p)
        if rec is None:
            print(f"  Phase {p}: no records.json found, skipping")
            continue
        all_recs[p] = rec
        print(f"\n=== Phase {p} ({rec.get('n_params_M', '?')}M params) ===")

        suffix = f"_phase{p}"
        plot_E1(rec, output_dir, suffix)
        plot_E2(rec, output_dir, suffix)
        plot_E3(rec, output_dir, suffix)
        plot_main_figure(rec, output_dir, suffix)

    if len(all_recs) > 1:
        print("\n=== Cross-phase comparison ===")
        plot_cross_phase(all_recs, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
