"""
Publication-quality re-plot of Claim 1 evidence.
Style: mirrors acl_arr_march_gac_tensorboard/plot_figures.py
  - serif / Times New Roman font
  - TensorBoard dual-layer: thin raw + thick EMA-smoothed with markers
  - No top/right spines
  - Colorblind-friendly palette
  - 600 dpi save
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Style (mirrors reference) ─────────────────────────────────────────────────
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

# Colorblind-friendly palette, one color per CMS level
LEVEL_META = {
    'cms_c1':   dict(color='#C0392B', ls='-',  mk='o',  label=r'$C\!=\!1$ (fast)'),
    'cms_c4':   dict(color='#2980B9', ls='--', mk='s',  label=r'$C\!=\!4$'),
    'cms_c32':  dict(color='#27AE60', ls='-.', mk='^',  label=r'$C\!=\!32$'),
    'cms_c128': dict(color='#8E44AD', ls=':',  mk='D',  label=r'$C\!=\!128$ (slow)'),
}
THEORY_COLOR = '#555555'

DRAW_ORDER = ['cms_c1', 'cms_c4', 'cms_c32', 'cms_c128']


# ── Utilities ─────────────────────────────────────────────────────────────────

def ema_smooth(data: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """TensorBoard-style EMA smoothing."""
    out = np.empty_like(data, dtype=float)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * data[i]
    return out


def style_axis(ax, xlabel: str = 'Task B Training Steps',
               ylabel: str = '', title: str = ''):
    ax.set_xlabel(xlabel, fontweight='bold', labelpad=4)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=4)
    if title:
        ax.set_title(title, fontweight='bold', pad=6)
    ax.grid(True, color='#CCCCCC', linestyle='-', linewidth=0.4, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='major', length=3, width=0.6)


def plot_level(ax, steps: np.ndarray, values: np.ndarray, name: str,
               smooth_alpha: float = 0.85, marker_every: int = 60,
               is_primary: bool = False, zorder_base: int = 2):
    """Dual-layer TensorBoard-style plot for one CMS level."""
    m = LEVEL_META[name]
    color, ls, mk, label = m['color'], m['ls'], m['mk'], m['label']
    zo = zorder_base + (4 if is_primary else 0)
    smoothed = ema_smooth(values, smooth_alpha)

    # Layer 1: raw — thin, faint
    ax.plot(steps, values, color=color, alpha=0.15, linewidth=0.7, zorder=zo - 1)
    # Layer 2: smoothed — bold, markers
    ax.plot(steps, smoothed, color=color, linestyle=ls,
            linewidth=2.0 if is_primary else 1.5,
            label=label, zorder=zo,
            marker=mk, markersize=4.0 if is_primary else 3.2,
            markevery=marker_every,
            markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.0)


def add_legend(ax, ncol: int = 2, loc: str = 'best'):
    leg = ax.legend(loc=loc, ncol=ncol, frameon=True, fancybox=False,
                    edgecolor='#AAAAAA', framealpha=0.95,
                    borderpad=0.4, handlelength=2.2, columnspacing=1.0,
                    handletextpad=0.5)
    leg.get_frame().set_linewidth(0.6)


# ── Load data ─────────────────────────────────────────────────────────────────

output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./results')
with open(output_dir / 'records.json') as f:
    rec = json.load(f)

steps = np.array(rec['step'])
beta = 0.9
n_levels = 4
level_names = ['cms_c1', 'cms_c4', 'cms_c32', 'cms_c128']
update_periods = [1, 4, 32, 128]

# Normalise projection series
proj_norm = {}
for n in level_names:
    raw = np.array(rec[f'mom_proj_norm_{n}'])
    proj_norm[n] = raw

proj_abs = {n: np.array(rec[f'mom_proj_abs_{n}']) for n in level_names}
drift = {n: np.array(rec[f'param_drift_{n}']) for n in level_names}


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1  ── Momentum projection decay  (2 panels)
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel (a): wall-clock steps, all levels + theory
ax = axes[0]
for name in DRAW_ORDER:
    vals = proj_norm[name]
    valid = ~np.isnan(vals)
    primary = (name == 'cms_c1')
    plot_level(ax, steps[valid], vals[valid], name,
               smooth_alpha=0.80, marker_every=55, is_primary=primary)

# Theory β^t overlay
theory = beta ** np.arange(len(steps))
ax.plot(steps, theory, color=THEORY_COLOR, ls='--', lw=1.3, alpha=0.7,
        label=rf'$\beta^t$ (theory, $\beta={beta}$)', zorder=1)

style_axis(ax,
           xlabel='Task $B$ Training Steps',
           ylabel=r'Normalised $\|\mathbf{U}_A^\top \mathbf{m}_t\|$',
           title=r'(a) Momentum Projection onto Old-Task Subspace')
add_legend(ax, ncol=2, loc='upper right')
ax.set_ylim(bottom=0)

# Panel (b): effective-update axis — all levels should overlap
ax = axes[1]
for name, C in zip(DRAW_ORDER, update_periods):
    vals = proj_norm[name]
    valid = ~np.isnan(vals)
    k_eff = np.array([s // C + 1 for s in steps])
    primary = (name == 'cms_c128')  # slowest is "primary" here
    plot_level(ax, k_eff[valid], vals[valid], name,
               smooth_alpha=0.75, marker_every=8, is_primary=primary)
    # Theory per level
    ax.plot(k_eff, beta ** k_eff, color=LEVEL_META[name]['color'],
            ls=':', lw=0.8, alpha=0.35, zorder=1)

style_axis(ax,
           xlabel=r'Effective Updates $k = \lfloor t/C \rfloor$',
           ylabel=r'Normalised $\|\mathbf{U}_A^\top \mathbf{m}_t\|$',
           title=r'(b) Decay Aligned to Effective Updates')
add_legend(ax, ncol=2, loc='upper right')
ax.set_ylim(bottom=0)

fig.suptitle(
    r'Fig. 1 — Gradient Memory Collapse: momentum projection '
    r'onto old-task subspace $\mathbf{U}_A$',
    fontsize=12, y=1.01, fontweight='bold',
)
plt.tight_layout(h_pad=1.5, w_pad=2.0)
fig.savefig(output_dir / 'pub_fig1_momentum_collapse.pdf')
fig.savefig(output_dir / 'pub_fig1_momentum_collapse.png')
print('Saved pub_fig1')
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2  ── Optimizer-state collapse vs parameter forgetting  (2×2)
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(11, 7.0))

for idx, (name, C) in enumerate(zip(level_names, update_periods)):
    ax = axes[idx // 2][idx % 2]
    p = proj_norm[name]
    dr = drift[name]
    valid = ~np.isnan(p) & ~np.isnan(dr)
    s_v = steps[valid]

    ax2 = ax.twinx()

    # Momentum projection (left y)
    raw_p = p[valid]
    smooth_p = ema_smooth(raw_p, 0.80)
    ax.plot(s_v, raw_p, color='#C0392B', alpha=0.15, lw=0.7)
    ax.plot(s_v, smooth_p, color='#C0392B', lw=2.0, ls='-',
            marker='o', markersize=3.5, markevery=55,
            markerfacecolor='white', markeredgecolor='#C0392B', markeredgewidth=0.9,
            label='Mom.\ proj.\ (left)', zorder=4)

    # Param drift (right y)
    raw_d = dr[valid]
    smooth_d = ema_smooth(raw_d, 0.80)
    ax2.plot(s_v, raw_d, color='#2980B9', alpha=0.15, lw=0.7)
    ax2.plot(s_v, smooth_d, color='#2980B9', lw=1.8, ls='--',
             marker='s', markersize=3.0, markevery=55,
             markerfacecolor='white', markeredgecolor='#2980B9', markeredgewidth=0.9,
             label='Param drift (right)', zorder=3)

    ax.tick_params(axis='y', labelcolor='#C0392B')
    ax2.tick_params(axis='y', labelcolor='#2980B9')
    ax.set_ylabel(r'Norm.\ momentum projection', color='#C0392B',
                  fontweight='bold', labelpad=3, fontsize=10)
    ax2.set_ylabel(r'Relative param.\ drift $\|\Delta\theta\|/\|\theta_A\|$',
                   color='#2980B9', fontweight='bold', labelpad=3, fontsize=10)
    ax.set_xlabel('Task $B$ Steps', fontweight='bold', labelpad=3, fontsize=10)

    title_str = rf'({chr(97+idx)}) Level $C\!=\!{C}$'
    ax.set_title(title_str, fontweight='bold', pad=6)
    ax.grid(True, color='#CCCCCC', linewidth=0.4, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.tick_params(axis='both', length=3, width=0.6)
    ax2.tick_params(axis='both', length=3, width=0.6)

    # Combined legend
    h1 = Line2D([0], [0], color='#C0392B', lw=2, ls='-',
                marker='o', markersize=3.5, markerfacecolor='white',
                label='Mom.\ projection')
    h2 = Line2D([0], [0], color='#2980B9', lw=1.8, ls='--',
                marker='s', markersize=3.0, markerfacecolor='white',
                label='Param.\ drift')
    ax.legend(handles=[h1, h2], loc='center right', fontsize=8.5,
              frameon=True, fancybox=False, edgecolor='#AAAAAA',
              framealpha=0.95).get_frame().set_linewidth(0.6)

fig.suptitle(
    r'Fig. 2 — Optimizer-state collapse precedes '
    r'parameter forgetting at every CMS level',
    fontsize=12, y=1.01, fontweight='bold',
)
plt.tight_layout(h_pad=2.0, w_pad=2.0)
fig.savefig(output_dir / 'pub_fig2_collapse_vs_forgetting.pdf')
fig.savefig(output_dir / 'pub_fig2_collapse_vs_forgetting.png')
print('Saved pub_fig2')
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3  ── No differentiated decay: all levels at same β per eff. update
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel (a): wall-clock — slow levels appear persistent
ax = axes[0]
for name in DRAW_ORDER:
    vals = proj_norm[name]
    valid = ~np.isnan(vals)
    plot_level(ax, steps[valid], vals[valid], name,
               smooth_alpha=0.80, marker_every=55, is_primary=(name == 'cms_c128'))
style_axis(ax,
           xlabel='Task $B$ Steps (wall-clock)',
           ylabel=r'Normalised $\|\mathbf{U}_A^\top \mathbf{m}_t\|$',
           title='(a) Apparent Persistence — slow levels seem stable')
add_legend(ax, ncol=2, loc='upper right')
ax.set_ylim(bottom=0)

# Panel (b): step-wise decay ratio → all cluster around β = 0.9
ax = axes[1]
ax.axhline(y=beta, color=THEORY_COLOR, ls='--', lw=1.4, alpha=0.8,
           label=rf'$\beta = {beta}$', zorder=1)

for name, C in zip(DRAW_ORDER, update_periods):
    vals = proj_norm[name]
    valid = ~np.isnan(vals)
    s_v = steps[valid]
    y_v = vals[valid]
    # Ratio at actual update steps
    prev = np.roll(y_v, 1); prev[0] = y_v[0]
    ratio = y_v / np.maximum(prev, 1e-10)
    update_mask = np.array([int(s_v[j]) % C == 0 for j in range(len(s_v))])
    k_u = np.array([s_v[j] // C + 1 for j in range(len(s_v))])[update_mask]
    r_u = ratio[update_mask]
    # Only show first 80 effective updates
    keep = k_u <= 80
    if keep.sum() > 2:
        m = LEVEL_META[name]
        ax.scatter(k_u[keep], r_u[keep], color=m['color'],
                   alpha=0.55, s=22, zorder=3,
                   label=rf'$C\!=\!{C}$')

style_axis(ax,
           xlabel=r'Effective Update Index $k$',
           ylabel=r'Step-wise ratio $r_k / r_{k-1}$',
           title=rf'(b) Per-step decay $\approx\beta\!=\!{beta}$ (identical across levels)')
add_legend(ax, ncol=2, loc='upper right')
ax.set_ylim(0.45, 1.35)

fig.suptitle(
    r'Fig. 3 — Vanilla HOPE has no differentiated decay: '
    r'all CMS levels lose gradient memory at the same rate per effective update',
    fontsize=12, y=1.01, fontweight='bold',
)
plt.tight_layout(h_pad=1.5, w_pad=2.0)
fig.savefig(output_dir / 'pub_fig3_decay_comparison.pdf')
fig.savefig(output_dir / 'pub_fig3_decay_comparison.png')
print('Saved pub_fig3')
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4  ── Summary panel  (2×3 grid)
# ═══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(15, 8.5))
gs = gridspec.GridSpec(2, 3, hspace=0.55, wspace=0.38)

# ── (a) E1: projection decay ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
for name in DRAW_ORDER:
    vals = proj_norm[name]
    valid = ~np.isnan(vals)
    plot_level(ax, steps[valid], vals[valid], name,
               smooth_alpha=0.80, marker_every=60, is_primary=(name == 'cms_c1'))
theory_y = beta ** np.arange(len(steps))
ax.plot(steps, theory_y, color=THEORY_COLOR, ls='--', lw=1.2, alpha=0.7,
        label=rf'$\beta^t$', zorder=1)
style_axis(ax, ylabel=r'Norm. $\|\mathbf{U}_A^\top\mathbf{m}_t\|$',
           title=r'(a) $\mathbf{E_1}$: Momentum projection decay')
add_legend(ax, ncol=1, loc='upper right')
ax.set_ylim(bottom=0)

# ── (b) E2 fast level ─────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
name, C = 'cms_c1', 1
p_v = proj_norm[name]; d_v = drift[name]
valid = ~np.isnan(p_v) & ~np.isnan(d_v)
s_v = steps[valid]
ax2 = ax.twinx()
sp = ema_smooth(p_v[valid], 0.80); sd = ema_smooth(d_v[valid], 0.80)
ax.plot(s_v, p_v[valid], color='#C0392B', alpha=0.15, lw=0.7)
ax.plot(s_v, sp, color='#C0392B', lw=2.0, ls='-',
        marker='o', markersize=3.5, markevery=55,
        markerfacecolor='white', markeredgecolor='#C0392B', markeredgewidth=0.9)
ax2.plot(s_v, d_v[valid], color='#2980B9', alpha=0.15, lw=0.7)
ax2.plot(s_v, sd, color='#2980B9', lw=1.8, ls='--',
         marker='s', markersize=3.0, markevery=55,
         markerfacecolor='white', markeredgecolor='#2980B9', markeredgewidth=0.9)
ax.tick_params(axis='y', labelcolor='#C0392B'); ax2.tick_params(axis='y', labelcolor='#2980B9')
ax.set_ylabel('Mom. proj.', color='#C0392B', fontweight='bold', fontsize=10)
ax2.set_ylabel('Param. drift', color='#2980B9', fontweight='bold', fontsize=10)
ax.set_xlabel('Task $B$ Steps', fontweight='bold', fontsize=10, labelpad=3)
ax.set_title(rf'(b) $\mathbf{{E_2}}$: Collapse vs forgetting ($C\!=\!{C}$)',
             fontweight='bold', pad=6)
ax.grid(True, color='#CCCCCC', linewidth=0.4, alpha=0.6)
ax.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)
h1 = Line2D([0],[0],color='#C0392B',lw=2,label='Mom. proj.')
h2 = Line2D([0],[0],color='#2980B9',lw=1.8,ls='--',label='Param. drift')
ax.legend(handles=[h1,h2],fontsize=8,frameon=True,fancybox=False,
          edgecolor='#AAAAAA',framealpha=0.95).get_frame().set_linewidth(0.6)

# ── (c) E2 slow level ─────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
name, C = 'cms_c128', 128
p_v = proj_norm[name]; d_v = drift[name]
valid = ~np.isnan(p_v) & ~np.isnan(d_v)
s_v = steps[valid]
ax2 = ax.twinx()
sp = ema_smooth(p_v[valid], 0.80); sd = ema_smooth(d_v[valid], 0.80)
ax.plot(s_v, p_v[valid], color='#8E44AD', alpha=0.15, lw=0.7)
ax.plot(s_v, sp, color='#8E44AD', lw=2.0, ls='-',
        marker='D', markersize=3.5, markevery=55,
        markerfacecolor='white', markeredgecolor='#8E44AD', markeredgewidth=0.9)
ax2.plot(s_v, d_v[valid], color='#2980B9', alpha=0.15, lw=0.7)
ax2.plot(s_v, sd, color='#2980B9', lw=1.8, ls='--',
         marker='s', markersize=3.0, markevery=55,
         markerfacecolor='white', markeredgecolor='#2980B9', markeredgewidth=0.9)
ax.tick_params(axis='y', labelcolor='#8E44AD'); ax2.tick_params(axis='y', labelcolor='#2980B9')
ax.set_ylabel('Mom. proj.', color='#8E44AD', fontweight='bold', fontsize=10)
ax2.set_ylabel('Param. drift', color='#2980B9', fontweight='bold', fontsize=10)
ax.set_xlabel('Task $B$ Steps', fontweight='bold', fontsize=10, labelpad=3)
ax.set_title(rf'(c) $\mathbf{{E_2}}$: Collapse vs forgetting ($C\!=\!{C}$)',
             fontweight='bold', pad=6)
ax.grid(True, color='#CCCCCC', linewidth=0.4, alpha=0.6)
ax.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)
h1 = Line2D([0],[0],color='#8E44AD',lw=2,label='Mom. proj.')
h2 = Line2D([0],[0],color='#2980B9',lw=1.8,ls='--',label='Param. drift')
ax.legend(handles=[h1,h2],fontsize=8,frameon=True,fancybox=False,
          edgecolor='#AAAAAA',framealpha=0.95).get_frame().set_linewidth(0.6)

# ── (d) E3: memory remaining bar ──────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
T_q = 200
rem_obs, rem_th = [], []
for name, C in zip(level_names, update_periods):
    y = proj_norm[name]
    rem = float(y[T_q]) if (T_q < len(y) and not np.isnan(y[T_q])) else 0.0
    rem_obs.append(rem)
    rem_th.append(beta ** (T_q // C + 1))
x = np.arange(n_levels); w = 0.35
colors4 = [LEVEL_META[n]['color'] for n in level_names]
bars1 = ax.bar(x - w/2, rem_obs, w, color=colors4, alpha=0.85, label='Observed')
bars2 = ax.bar(x + w/2, rem_th, w, color=colors4, alpha=0.38,
               hatch='//', edgecolor='black', lw=0.5, label=r'Theory $\beta^k$')
ax.set_xticks(x); ax.set_xticklabels([rf'$C\!=\!{C}$' for C in update_periods])
style_axis(ax, xlabel='CMS Level',
           ylabel='Fraction of projection remaining',
           title=rf'(d) $\mathbf{{E_3}}$: Memory at step {T_q}')
ax.legend(fontsize=8.5, frameon=True, fancybox=False,
          edgecolor='#AAAAAA', framealpha=0.95).get_frame().set_linewidth(0.6)
ax.set_ylim(0, 1.15)

# ── (e) Absolute momentum projection ─────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
for name in DRAW_ORDER:
    vals = proj_abs[name]; valid = ~np.isnan(vals)
    plot_level(ax, steps[valid], vals[valid], name,
               smooth_alpha=0.80, marker_every=60, is_primary=(name == 'cms_c1'))
style_axis(ax, ylabel=r'$\|\mathbf{U}_A^\top \mathbf{m}_t\|$ (absolute)',
           title=r'(e) Absolute momentum projection')
add_legend(ax, ncol=2, loc='upper right')

# ── (f) Half-life comparison ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
emp_hl, theory_hl = [], []
for name, C in zip(level_names, update_periods):
    y = proj_norm[name]; s = steps
    valid = ~np.isnan(y); y_v = y[valid]; s_v = s[valid]
    hl = float('nan')
    for j in range(1, len(y_v)):
        if y_v[j] <= 0.5:
            f = (y_v[j-1] - 0.5) / max(y_v[j-1] - y_v[j], 1e-12)
            hl = float(s_v[j-1] + f * (s_v[j] - s_v[j-1]))
            break
    emp_hl.append(hl if not math.isnan(hl) else 0)
    theory_hl.append(math.log(0.5) / math.log(beta) * C)

x = np.arange(n_levels)
bars1 = ax.bar(x - w/2, emp_hl, w, color=colors4, alpha=0.85, label='Observed')
bars2 = ax.bar(x + w/2, theory_hl, w, color=colors4, alpha=0.38,
               hatch='//', edgecolor='black', lw=0.5, label=r'Theory $k_{1/2}\cdot C$')
ax.set_xticks(x); ax.set_xticklabels([rf'$C\!=\!{C}$' for C in update_periods])
style_axis(ax, xlabel='CMS Level',
           ylabel='Half-life (steps)',
           title=r'(f) Momentum memory half-life')
ax.legend(fontsize=8.5, frameon=True, fancybox=False,
          edgecolor='#AAAAAA', framealpha=0.95).get_frame().set_linewidth(0.6)

fig.suptitle(
    r'Fig. 4 — Summary: Claim 1 Evidence — Gradient Memory Collapse is '
    r'a real optimizer-state problem in vanilla HOPE',
    fontsize=12, y=1.01, fontweight='bold',
)
fig.savefig(output_dir / 'pub_fig4_summary.pdf')
fig.savefig(output_dir / 'pub_fig4_summary.png')
print('Saved pub_fig4')
plt.close(fig)

print('\n=== All publication figures saved. ===')
