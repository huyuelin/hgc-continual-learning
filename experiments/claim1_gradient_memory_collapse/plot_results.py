"""Re-generate all plots from saved records.json (run after experiment completes)."""
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

# ── read data ────────────────────────────────────────────────────────────────
output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./results")
records_path = output_dir / "records.json"
with open(records_path) as f:
    records = json.load(f)

# Reconstruct metadata from record keys
steps = np.array(records["step"])
# e.g. keys like "mom_proj_cms_c1" → extract period
level_names = []
update_periods = []
for k in records:
    if k.startswith("mom_proj_cms_c") and "norm" not in k and "abs" not in k:
        name = k[len("mom_proj_"):]
        c = int(name.replace("cms_c", ""))
        level_names.append(name)
        update_periods.append(c)
# Sort by period
paired = sorted(zip(update_periods, level_names))
update_periods = [p[0] for p in paired]
level_names = [p[1] for p in paired]
n_levels = len(level_names)
beta = 0.9
task_b_steps = int(steps[-1]) + 1

LEVEL_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8"]
LEVEL_LSTYLE = ["-", "--", "-.", ":"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 150,
})

# ── Figure 1: Momentum projection norm decay ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax1, ax2 = axes

for i, (name, C) in enumerate(zip(level_names, update_periods)):
    y = np.array(records[f"mom_proj_norm_{name}"])
    valid = ~np.isnan(y)
    if valid.sum() > 0:
        ax1.plot(steps[valid], y[valid], color=LEVEL_COLORS[i],
                 ls=LEVEL_LSTYLE[i], lw=2, label=f"Level C={C}")

theory_x = np.arange(len(steps))
theory_y = beta ** theory_x
ax1.plot(theory_x, theory_y, "k--", lw=1.5, alpha=0.5, label=f"Theoretical β^t (β={beta})")
ax1.set_xlabel("Task B Training Steps")
ax1.set_ylabel("Normalized Projection Norm (relative to step 0)")
ax1.set_title("(a) Momentum Projection onto Old-Task Subspace\n(Vanilla HOPE, normalized)")
ax1.legend(framealpha=0.9)
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3)

for i, (name, C) in enumerate(zip(level_names, update_periods)):
    effective_steps = np.array([s // C + 1 for s in steps])
    theory_y_l = beta ** effective_steps
    y = np.array(records[f"mom_proj_norm_{name}"])
    valid = ~np.isnan(y)
    if valid.sum() > 0:
        ax2.plot(effective_steps[valid], y[valid], color=LEVEL_COLORS[i],
                 ls=LEVEL_LSTYLE[i], lw=2, label=f"Observed C={C}")
        ax2.plot(effective_steps, theory_y_l, color=LEVEL_COLORS[i],
                 ls=":", lw=1.2, alpha=0.5, label=f"Theory β^k C={C}")

ax2.set_xlabel("Effective Updates on Task B (k = step/C)")
ax2.set_ylabel("Normalized Projection Norm")
ax2.set_title("(b) Decay vs Effective Updates\n(All levels follow β^k regardless of C)")
ax2.legend(framealpha=0.9, ncol=2, fontsize=8)
ax2.set_ylim(bottom=0)
ax2.grid(True, alpha=0.3)

fig.suptitle("Figure 1: Gradient Memory Collapse in Vanilla HOPE\n"
             "Momentum Buffer Projection onto Task A Subspace U_A", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(output_dir / "fig1_momentum_collapse.pdf", bbox_inches="tight")
fig.savefig(output_dir / "fig1_momentum_collapse.png", bbox_inches="tight")
print("Saved fig1")
plt.close(fig)

# ── Figure 2: Optimizer-state collapse vs parameter forgetting ───────────────
fig, axes = plt.subplots(1, n_levels, figsize=(4.5 * n_levels, 4.5), sharey=False)
if n_levels == 1:
    axes = [axes]

for i, (name, C, ax) in enumerate(zip(level_names, update_periods, axes)):
    proj = np.array(records[f"mom_proj_norm_{name}"])
    drift = np.array(records[f"param_drift_{name}"])
    valid = ~np.isnan(proj) & ~np.isnan(drift)
    ax2t = ax.twinx()
    l1, = ax.plot(steps[valid], proj[valid], color="#e41a1c", lw=2, label="Mom. Proj.")
    l2, = ax2t.plot(steps[valid], drift[valid], color="#377eb8", ls="--", lw=2, label="Param Drift")
    ax.set_xlabel("Task B Steps")
    ax.set_ylabel("Norm. Momentum Projection", color="#e41a1c")
    ax2t.set_ylabel("Relative Param Drift", color="#377eb8")
    ax.tick_params(axis="y", labelcolor="#e41a1c")
    ax2t.tick_params(axis="y", labelcolor="#377eb8")
    ax.set_title(f"Level C={C}")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[l1, l2], loc="upper right", fontsize=8)

fig.suptitle("Figure 2: Optimizer-State Collapse vs Parameter Forgetting\n"
             "Momentum projection decays faster / earlier than parameter drift",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(output_dir / "fig2_collapse_vs_forgetting.pdf", bbox_inches="tight")
fig.savefig(output_dir / "fig2_collapse_vs_forgetting.png", bbox_inches="tight")
print("Saved fig2")
plt.close(fig)

# ── Figure 3: Same decay rate per effective update across all levels ──────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax_left, ax_right = axes

for i, (name, C) in enumerate(zip(level_names, update_periods)):
    y = np.array(records[f"mom_proj_norm_{name}"])
    valid = ~np.isnan(y)
    if valid.sum() > 0:
        ax_left.plot(steps[valid], y[valid], color=LEVEL_COLORS[i],
                     ls=LEVEL_LSTYLE[i], lw=2.5, label=f"C={C}")

ax_left.set_xlabel("Task B Training Steps (wall-clock)")
ax_left.set_ylabel("Normalized Momentum Projection")
ax_left.set_title("(a) Apparent Persistence by Level\n(slow levels seem more stable in wall-clock time)")
ax_left.legend(title="Update Period C")
ax_left.grid(True, alpha=0.3)
ax_left.set_ylim(bottom=0)

ax_right.axhline(y=beta, color="gray", ls="--", alpha=0.5, lw=1.5, label=f"β = {beta}")
for i, (name, C) in enumerate(zip(level_names, update_periods)):
    y = np.array(records[f"mom_proj_norm_{name}"])
    valid = ~np.isnan(y)
    if valid.sum() > 2:
        k = np.array([s // C + 1 for s in steps])
        prev_y = np.roll(y[valid], 1)
        prev_y[0] = y[valid][0]
        ratio = y[valid] / np.maximum(prev_y, 1e-10)
        update_mask = np.array([int(steps[np.where(valid)[0][j]]) % C == 0
                                 for j in range(valid.sum())])
        k_u = k[valid][update_mask]
        r_u = ratio[update_mask]
        if len(k_u) > 1:
            ax_right.scatter(k_u[:80], r_u[:80], color=LEVEL_COLORS[i],
                             alpha=0.6, s=25, label=f"C={C}")

ax_right.set_xlabel("Effective Update Index k")
ax_right.set_ylabel("Step-wise Projection Ratio (r_k / r_{k-1})")
ax_right.set_title(f"(b) Per-Step Decay Rate ≈ β = {beta}\n(identical across all levels → no hierarchy)")
ax_right.set_ylim(0.4, 1.3)
ax_right.legend()
ax_right.grid(True, alpha=0.3)

fig.suptitle("Figure 3: Vanilla HOPE Has No Differentiated Decay\n"
             "All CMS Levels Lose Gradient Memory at the Same Rate per Effective Update",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(output_dir / "fig3_decay_comparison.pdf", bbox_inches="tight")
fig.savefig(output_dir / "fig3_decay_comparison.png", bbox_inches="tight")
print("Saved fig3")
plt.close(fig)

# ── Figure 4: Summary panel ───────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(2, 3, hspace=0.5, wspace=0.4)

# Row 0, col 0: E1 — momentum projection normalized
ax = fig.add_subplot(gs[0, 0])
for i, (name, C) in enumerate(zip(level_names, update_periods)):
    y = np.array(records[f"mom_proj_norm_{name}"])
    valid = ~np.isnan(y)
    if valid.sum() > 0:
        ax.plot(steps[valid], y[valid], color=LEVEL_COLORS[i],
                ls=LEVEL_LSTYLE[i], lw=2, label=f"C={C}")
t_theory = beta ** np.arange(len(steps))
ax.plot(np.arange(len(steps)), t_theory, "k--", lw=1.2, alpha=0.5, label=f"β^t")
ax.set_title("E1: Momentum Projection Decay")
ax.set_xlabel("Task B Steps")
ax.set_ylabel("Norm. Proj. (relative)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# Row 0, col 1 & 2: E2 — fastest and slowest level
for col_offset, level_idx in enumerate([0, n_levels - 1]):
    ax = fig.add_subplot(gs[0, 1 + col_offset])
    name = level_names[level_idx]
    C = update_periods[level_idx]
    proj = np.array(records[f"mom_proj_norm_{name}"])
    drift = np.array(records[f"param_drift_{name}"])
    valid = ~np.isnan(proj) & ~np.isnan(drift)
    ax2t = ax.twinx()
    ax.plot(steps[valid], proj[valid], color="#e41a1c", lw=2, label="Mom. Proj.")
    ax2t.plot(steps[valid], drift[valid], color="#377eb8", ls="--", lw=2, label="Param Drift")
    ax.set_title(f"E2: Collapse vs Forgetting\nLevel C={C}")
    ax.set_xlabel("Task B Steps")
    ax.set_ylabel("Mom. Proj.", color="#e41a1c")
    ax2t.set_ylabel("Param Drift", color="#377eb8")
    ax.tick_params(axis="y", labelcolor="#e41a1c")
    ax2t.tick_params(axis="y", labelcolor="#377eb8")
    ax.grid(True, alpha=0.3)
    lines = [Line2D([0], [0], color="#e41a1c", lw=2, label="Mom. Proj."),
             Line2D([0], [0], color="#377eb8", lw=2, ls="--", label="Param Drift")]
    ax.legend(handles=lines, fontsize=8)

# Row 1, col 0: remaining memory at step 200
ax = fig.add_subplot(gs[1, 0])
T_query = min(200, task_b_steps - 1)
remaining_proj, remaining_theory = [], []
for i, (name, C) in enumerate(zip(level_names, update_periods)):
    y = np.array(records[f"mom_proj_norm_{name}"])
    idx = np.where(~np.isnan(y))[0]
    rem = float(y[T_query]) if (len(idx) > 0 and T_query < len(y) and not np.isnan(y[T_query])) else float("nan")
    remaining_proj.append(rem)
    k_eff = T_query // C + 1
    remaining_theory.append(beta ** k_eff)

x = np.arange(n_levels)
width = 0.35
valid_rp = [v if not math.isnan(v) else 0 for v in remaining_proj]
ax.bar(x - width/2, valid_rp, width, color=LEVEL_COLORS[:n_levels], alpha=0.85, label="Observed")
ax.bar(x + width/2, remaining_theory, width, color=LEVEL_COLORS[:n_levels], alpha=0.4,
       hatch="//", edgecolor="black", label="Theory β^k")
ax.set_xticks(x)
ax.set_xticklabels([f"C={C}" for C in update_periods])
ax.set_xlabel("CMS Level")
ax.set_ylabel("Fraction of Projection Remaining")
ax.set_title(f"E3: Memory Remaining at Step {T_query}\n(Observed ≈ Theory)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# Row 1, col 1: absolute momentum projection
ax = fig.add_subplot(gs[1, 1])
for i, (name, C) in enumerate(zip(level_names, update_periods)):
    y = np.array(records[f"mom_proj_abs_{name}"])
    valid = ~np.isnan(y)
    if valid.sum() > 0:
        ax.plot(steps[valid], y[valid], color=LEVEL_COLORS[i],
                ls=LEVEL_LSTYLE[i], lw=2, label=f"C={C}")
ax.set_title("||U_A^T m_t|| Absolute")
ax.set_xlabel("Task B Steps")
ax.set_ylabel("||U_A^T m_t||")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Row 1, col 2: half-life comparison
ax = fig.add_subplot(gs[1, 2])
emp_half_lives, theory_half_lives = [], []
for i, (name, C) in enumerate(zip(level_names, update_periods)):
    y = np.array(records[f"mom_proj_norm_{name}"])
    valid = ~np.isnan(y)
    s_v = steps[valid]
    y_v = y[valid]
    hl_emp = float("nan")
    for j in range(1, len(y_v)):
        if y_v[j] <= 0.5:
            frac = (y_v[j-1] - 0.5) / (y_v[j-1] - y_v[j] + 1e-12)
            hl_emp = float(s_v[j-1] + frac * (s_v[j] - s_v[j-1]))
            break
    k_half = math.log(0.5) / math.log(beta)
    hl_theory = k_half * C
    emp_half_lives.append(hl_emp)
    theory_half_lives.append(hl_theory)

x = np.arange(n_levels)
valid_emp = [v if not math.isnan(v) else 0 for v in emp_half_lives]
ax.bar(x - width/2, valid_emp, width, color=LEVEL_COLORS[:n_levels], alpha=0.85, label="Observed")
ax.bar(x + width/2, theory_half_lives, width, color=LEVEL_COLORS[:n_levels], alpha=0.4,
       hatch="//", edgecolor="black", label="Theory k·C")
ax.set_xticks(x)
ax.set_xticklabels([f"C={C}" for C in update_periods])
ax.set_xlabel("CMS Level")
ax.set_ylabel("Half-life (steps)")
ax.set_title("Momentum Memory Half-Life\n(Slow levels only persist due to fewer updates)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Claim 1 Evidence: Gradient Memory Collapse in Vanilla HOPE\n"
             "Optimizer-state level memory loss is real, precedes parameter forgetting,\n"
             "and shows no intrinsic frequency hierarchy",
             fontsize=13, y=1.01)
fig.savefig(output_dir / "fig4_summary_panel.pdf", bbox_inches="tight")
fig.savefig(output_dir / "fig4_summary_panel.png", bbox_inches="tight")
print("Saved fig4")
plt.close(fig)

print("=== All figures regenerated. ===")
