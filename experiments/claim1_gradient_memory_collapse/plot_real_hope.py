"""
Publication-quality figures for Claim 1 evidence from REAL HOPE experiment.

Reads records.json produced by probe_real_hope.py and generates:
  Fig 1 (E1): Momentum projection decay per CMS level
  Fig 2 (E2): Optimizer-state collapse vs parameter forgetting
  Fig 3 (E3): Decay rate comparison across levels (no hierarchy in vanilla HOPE)

Style: Times New Roman, TensorBoard dual-layer, no top/right spines, 600 dpi.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Style ─────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.6,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Level metadata: real HOPE naming
LEVEL_META = {
    'cms_fast':  dict(C=1,   color='#C0392B', ls='-',  mk='o',  label=r'$C\!=\!1$ (fast)'),
    'cms_mid':   dict(C=4,   color='#2980B9', ls='--', mk='s',  label=r'$C\!=\!4$'),
    'cms_slow':  dict(C=32,  color='#27AE60', ls='-.', mk='^',  label=r'$C\!=\!32$'),
    'cms_ultra': dict(C=128, color='#8E44AD', ls=':',  mk='D',  label=r'$C\!=\!128$ (slow)'),
}
THEORY_COLOR = '#555555'
DRAW_ORDER = ['cms_fast', 'cms_mid', 'cms_slow', 'cms_ultra']


def ema_smooth(data: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    out = np.empty_like(data, dtype=float)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * out[i-1] + (1 - alpha) * data[i]
    return out


def clean(arr):
    """Replace None with nan."""
    return np.array([float('nan') if v is None else float(v) for v in arr])


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_markers(ax, x, y, meta, every=40, ms=5):
    valid = ~np.isnan(y)
    x_v, y_v = x[valid], y[valid]
    idx = np.arange(0, len(x_v), every)
    ax.plot(x_v[idx], y_v[idx], marker=meta['mk'], color=meta['color'],
            ls='none', ms=ms, mew=0.8, mfc='white', zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
def main(records_path: str, output_dir: str):
    with open(records_path) as f:
        rec = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps = np.array(rec['step'], dtype=float)
    beta = 0.9

    # Compute normalized projection (relative to step-0 value)
    for ln in DRAW_ORDER:
        key = f"mom_proj_{ln}"
        key_norm = f"mom_proj_norm_{ln}"
        if key in rec and key_norm not in rec:
            arr = clean(rec[key])
            v0 = arr[0] if (len(arr) > 0 and not np.isnan(arr[0])) else 1.0
            rec[key_norm] = (arr / v0 if v0 > 1e-8 else arr).tolist()

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 1 (E1): Momentum Projection Decay
    # ══════════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # --- Left panel: wall-clock decay ---
    for ln in DRAW_ORDER:
        meta = LEVEL_META[ln]
        key = f"mom_proj_norm_{ln}"
        if key not in rec:
            continue
        y = clean(rec[key])
        valid = ~np.isnan(y)
        if valid.sum() < 3:
            continue
        y_sm = ema_smooth(y[valid], 0.88)
        ax1.plot(steps[valid], y[valid], color=meta['color'], alpha=0.25, lw=0.8)
        ax1.plot(steps[valid], y_sm, color=meta['color'], ls=meta['ls'], lw=2.0,
                 label=meta['label'])
        add_markers(ax1, steps[valid], y_sm, meta, every=50)

    # Theoretical β^t
    t_theory = beta ** np.arange(len(steps))
    ax1.plot(steps, t_theory, color=THEORY_COLOR, ls='--', lw=1.2, alpha=0.6,
             label=rf'$\beta^t$ ($\beta$={beta})')

    despine(ax1)
    ax1.set_xlabel('Task B Training Step')
    ax1.set_ylabel(r'$\| \mathbf{U}_A^\top \mathbf{m}_t \| / \| \mathbf{U}_A^\top \mathbf{m}_0 \|$')
    ax1.set_title('(a) Momentum Projection Decay (wall-clock)')
    ax1.legend(loc='upper right', framealpha=0.92, edgecolor='none')
    ax1.set_ylim(bottom=-0.05, top=1.15)
    ax1.grid(True, alpha=0.25)

    # --- Right panel: decay vs effective updates ---
    for ln in DRAW_ORDER:
        meta = LEVEL_META[ln]
        C = meta['C']
        key = f"mom_proj_norm_{ln}"
        if key not in rec:
            continue
        y = clean(rec[key])
        valid = ~np.isnan(y)
        if valid.sum() < 3:
            continue
        k_eff = np.array([s // C + 1 for s in steps], dtype=float)
        y_sm = ema_smooth(y[valid], 0.88)
        ax2.plot(k_eff[valid], y_sm, color=meta['color'], ls=meta['ls'], lw=2.0,
                 label=f'Obs. {meta["label"]}')
        # Theoretical
        theory = beta ** k_eff
        ax2.plot(k_eff, theory, color=meta['color'], ls=':', lw=1.0, alpha=0.5)

    despine(ax2)
    ax2.set_xlabel(r'Effective Updates $k = \lfloor t / C \rfloor + 1$')
    ax2.set_ylabel('Normalized Projection Norm')
    ax2.set_title(r'(b) Decay vs Effective Updates ($\approx \beta^k$ for all C)')
    ax2.legend(loc='upper right', framealpha=0.92, edgecolor='none', fontsize=8, ncol=2)
    ax2.set_ylim(bottom=-0.05, top=1.15)
    ax2.grid(True, alpha=0.25)

    fig.suptitle(
        'Evidence E1: Gradient Memory Collapse in Vanilla HOPE\n'
        r'Momentum buffer projection onto Task A subspace $\mathbf{U}_A$',
        fontsize=13, fontweight='bold', y=1.02,
    )
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(out / f'real_hope_E1_momentum_collapse.{ext}')
    print(f"Saved E1 figure")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 2 (E2): Optimizer-State Collapse vs Parameter Forgetting
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Only show levels with data
    active_levels = [ln for ln in DRAW_ORDER if f"mom_proj_norm_{ln}" in rec
                     and any(v is not None for v in rec[f"mom_proj_norm_{ln}"])]

    for idx, ln in enumerate(active_levels):
        if idx >= len(axes):
            break
        ax = axes[idx]
        meta = LEVEL_META[ln]
        C = meta['C']

        proj_key = f"mom_proj_norm_{ln}"
        drift_key = f"param_drift_{ln}"
        if proj_key not in rec or drift_key not in rec:
            ax.set_visible(False)
            continue

        proj = clean(rec[proj_key])
        drift = clean(rec[drift_key])
        valid = ~np.isnan(proj) & ~np.isnan(drift)
        if valid.sum() < 3:
            continue

        proj_sm = ema_smooth(proj[valid], 0.90)
        drift_sm = ema_smooth(drift[valid], 0.90)

        ax_right = ax.twinx()

        # Momentum projection (red, left axis)
        ax.plot(steps[valid], proj[valid], color='#C0392B', alpha=0.15, lw=0.6)
        l1, = ax.plot(steps[valid], proj_sm, color='#C0392B', lw=2.0,
                      label='Momentum Proj.')

        # Parameter drift (blue, right axis)
        ax_right.plot(steps[valid], drift[valid], color='#2980B9', alpha=0.15, lw=0.6)
        l2, = ax_right.plot(steps[valid], drift_sm, color='#2980B9', ls='--', lw=2.0,
                            label='Param. Drift')

        despine(ax)
        ax_right.spines['top'].set_visible(False)
        ax.set_xlabel('Task B Step')
        ax.set_ylabel('Momentum Proj. (norm.)', color='#C0392B')
        ax_right.set_ylabel('Param. Drift', color='#2980B9')
        ax.tick_params(axis='y', colors='#C0392B')
        ax_right.tick_params(axis='y', colors='#2980B9')
        ax.set_title(f'{meta["label"]}  (C={C})')
        ax.grid(True, alpha=0.2)
        ax.legend(handles=[l1, l2], loc='center right', fontsize=8,
                  framealpha=0.92, edgecolor='none')

    fig.suptitle(
        'Evidence E2: Optimizer-State Collapse Precedes Parameter Forgetting\n'
        'Momentum projection decays faster than parameter drift across all CMS levels',
        fontsize=13, fontweight='bold', y=1.01,
    )
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(out / f'real_hope_E2_collapse_vs_forgetting.{ext}')
    print(f"Saved E2 figure")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 3 (E3): Decay Rate Comparison — No Hierarchy in Vanilla HOPE
    # ══════════════════════════════════════════════════════════════════════════
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # --- Left: apparent persistence (wall-clock) ---
    for ln in DRAW_ORDER:
        meta = LEVEL_META[ln]
        key = f"mom_proj_norm_{ln}"
        if key not in rec:
            continue
        y = clean(rec[key])
        valid = ~np.isnan(y)
        if valid.sum() < 3:
            continue
        y_sm = ema_smooth(y[valid], 0.88)
        ax_l.plot(steps[valid], y[valid], color=meta['color'], alpha=0.2, lw=0.7)
        ax_l.plot(steps[valid], y_sm, color=meta['color'], ls=meta['ls'], lw=2.2,
                  label=meta['label'])

    despine(ax_l)
    ax_l.set_xlabel('Task B Training Step (wall-clock)')
    ax_l.set_ylabel('Normalized Momentum Projection')
    ax_l.set_title('(a) Apparent Persistence by Level\n(slow levels appear more stable)')
    ax_l.legend(title='Update Period', title_fontsize=9, framealpha=0.92, edgecolor='none')
    ax_l.set_ylim(bottom=-0.05)
    ax_l.grid(True, alpha=0.25)

    # --- Right: per-effective-update scatter + β reference line ---
    ax_r.axhline(y=beta, color='gray', ls='--', alpha=0.5, lw=1,
                 label=rf'$\beta = {beta}$')

    for ln in DRAW_ORDER:
        meta = LEVEL_META[ln]
        C = meta['C']
        key = f"mom_proj_norm_{ln}"
        if key not in rec:
            continue
        y = clean(rec[key])
        valid = ~np.isnan(y)
        if valid.sum() < 5:
            continue
        y_v = y[valid]
        s_v = steps[valid]

        # Compute ratio at effective update boundaries
        ratios_k = []
        ratios_v = []
        # Group by effective update index
        prev_k = -1
        prev_val = None
        for j in range(len(s_v)):
            k = int(s_v[j]) // C + 1
            if k != prev_k:
                if prev_val is not None and y_v[j] > 1e-8 and prev_val > 1e-8:
                    ratios_k.append(k)
                    ratios_v.append(y_v[j] / prev_val)
                prev_k = k
                prev_val = y_v[j]

        if len(ratios_k) > 1:
            ax_r.scatter(ratios_k[:100], ratios_v[:100],
                         color=meta['color'], alpha=0.55, s=22,
                         marker=meta['mk'], label=meta['label'],
                         edgecolors=meta['color'], linewidths=0.5)

    despine(ax_r)
    ax_r.set_xlabel(r'Effective Update Index $k$')
    ax_r.set_ylabel(r'Step-wise Ratio $r_k / r_{k-1}$')
    ax_r.set_title(rf'(b) Per-Step Decay Rate $\approx \beta = {beta}$'
                   '\n(identical across levels → no hierarchy)')
    ax_r.set_ylim(0.4, 1.25)
    ax_r.legend(framealpha=0.92, edgecolor='none', fontsize=8.5)
    ax_r.grid(True, alpha=0.25)

    fig.suptitle(
        'Evidence E3: Vanilla HOPE Has No Differentiated Decay Across Levels\n'
        'All CMS levels lose gradient memory at the same rate per effective update',
        fontsize=13, fontweight='bold', y=1.02,
    )
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(out / f'real_hope_E3_decay_comparison.{ext}')
    print(f"Saved E3 figure")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # Combined Summary Panel
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    # Panel 1: E1 momentum projection
    ax = fig.add_subplot(gs[0, 0])
    for ln in DRAW_ORDER:
        meta = LEVEL_META[ln]
        key = f"mom_proj_norm_{ln}"
        if key not in rec:
            continue
        y = clean(rec[key])
        valid = ~np.isnan(y)
        if valid.sum() < 3:
            continue
        y_sm = ema_smooth(y[valid], 0.88)
        ax.plot(steps[valid], y_sm, color=meta['color'], ls=meta['ls'], lw=2.0,
                label=meta['label'])
    t_theory = beta ** np.arange(len(steps))
    ax.plot(steps, t_theory, color=THEORY_COLOR, ls='--', lw=1.0, alpha=0.5,
            label=rf'$\beta^t$')
    despine(ax)
    ax.set_xlabel('Task B Step')
    ax.set_ylabel('Proj. Norm (normalized)')
    ax.set_title('E1: Momentum Projection Decay', fontweight='bold')
    ax.legend(fontsize=7.5, framealpha=0.92, edgecolor='none')
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=-0.05)

    # Panel 2: E2 fastest vs slowest
    ax = fig.add_subplot(gs[0, 1])
    for ln, lsty in [('cms_fast', '-'), ('cms_ultra', '--')]:
        meta = LEVEL_META[ln]
        proj_key = f"mom_proj_norm_{ln}"
        drift_key = f"param_drift_{ln}"
        if proj_key not in rec or drift_key not in rec:
            continue
        proj = clean(rec[proj_key])
        drift = clean(rec[drift_key])
        valid = ~np.isnan(proj) & ~np.isnan(drift)
        if valid.sum() < 3:
            continue
        proj_sm = ema_smooth(proj[valid], 0.90)
        drift_sm = ema_smooth(drift[valid], 0.90)
        ax.plot(steps[valid], proj_sm, color='#C0392B', ls=lsty, lw=2.0,
                label=f'Mom. {meta["label"]}')
        ax.plot(steps[valid], drift_sm, color='#2980B9', ls=lsty, lw=1.5,
                label=f'Drift {meta["label"]}')
    despine(ax)
    ax.set_xlabel('Task B Step')
    ax.set_ylabel('Normalized Value')
    ax.set_title('E2: Collapse vs Forgetting', fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.92, edgecolor='none')
    ax.grid(True, alpha=0.25)

    # Panel 3: E3 bar chart — remaining memory at step 200
    ax = fig.add_subplot(gs[0, 2])
    T_query = 200
    obs_vals = []
    theory_vals = []
    labels = []
    colors = []
    for ln in DRAW_ORDER:
        meta = LEVEL_META[ln]
        C = meta['C']
        key = f"mom_proj_norm_{ln}"
        if key not in rec:
            continue
        y = clean(rec[key])
        valid_idx = np.where(~np.isnan(y))[0]
        if len(valid_idx) > 0 and valid_idx[-1] >= T_query:
            obs_vals.append(y[T_query])
        else:
            obs_vals.append(float('nan'))
        k_eff = T_query // C + 1
        theory_vals.append(beta ** k_eff)
        labels.append(f'C={C}')
        colors.append(meta['color'])

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, obs_vals, w, color=colors, alpha=0.85, label='Observed')
    ax.bar(x + w/2, theory_vals, w, color=colors, alpha=0.35, hatch='//', label=r'Theory $\beta^k$')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    despine(ax)
    ax.set_xlabel('CMS Level')
    ax.set_ylabel('Fraction Remaining')
    ax.set_title(f'E3: Memory at Step {T_query}', fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.92, edgecolor='none')
    ax.grid(True, alpha=0.25, axis='y')

    fig.suptitle(
        'Claim 1 Summary: Gradient Memory Collapse in Real HOPE (DeepMomentum, nl_l2_precond)',
        fontsize=13, fontweight='bold', y=1.03,
    )
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(out / f'real_hope_summary.{ext}')
    print(f"Saved summary figure")
    plt.close(fig)

    print("\n=== All publication figures saved ===")


if __name__ == "__main__":
    records_path = sys.argv[1] if len(sys.argv) > 1 else "./results_real_hope/records.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./results_real_hope"
    main(records_path, output_dir)
