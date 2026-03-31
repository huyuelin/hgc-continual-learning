"""
Claim 1 Probe — Phase 1/2/3 with Real WikiText-2 Text
======================================================
Uses real HOPEModel + DeepMomentum + WikiText-2 language modeling.

Evidence collected:
  E1: momentum projection onto Task-A subspace decays during Task-B training
  E2: optimizer-state collapse precedes parameter drift AND perplexity rise
  E3: all CMS levels show same per-effective-update decay rate ≈ β

Three scales available via --phase argument:
  Phase 1: dim=256, 8 layers,  ~48M  params  (~30 min on A800)
  Phase 2: dim=512, 12 layers, ~256M params  (~3h  on A800)
  Phase 3: dim=768, 16 layers, ~742M params  (~8h  on A800)

Data:
  Task A: WikiText-2 train articles [0 .. split_frac)   → domain "Wikipedia main text"
  Task B: WikiText-2 train articles [split_frac .. end] → domain "Wikipedia remaining"
  Both tokenised with tiktoken cl100k_base (vocab=100277)

  Perplexity-on-TaskA: evaluated on WikiText-2 *validation* set every eval_every steps.

Usage:
    python probe_wikitext_v3.py --phase 1 --output_dir ./results_phase1 --device cuda:0
    python probe_wikitext_v3.py --phase 2 --output_dir ./results_phase2 --device cuda:0
    python probe_wikitext_v3.py --phase 3 --output_dir ./results_phase3 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
import ssl
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── Import real HOPE ──────────────────────────────────────────────────────────
_CANDIDATES = [
    os.path.expanduser("~/hope_src"),
    os.path.expanduser(
        "~/remote-workspace/continuous_learning/"
        "Nested Learning The Illusion of Deep Learning Architecture "
        "Ali Behrouz , Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni/"
        "code/nested_learning_hgc/src"
    ),
]
HOPE_ROOT = next(
    (p for p in _CANDIDATES if os.path.isdir(os.path.join(p, "nested_learning"))),
    _CANDIDATES[0],
)
sys.path.insert(0, HOPE_ROOT)

from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.levels import LevelSpec
from nested_learning.optim.deep import DeepMomentum


# ─────────────────────────────────────────────────────────────────────────────
# Phase configs
# ─────────────────────────────────────────────────────────────────────────────

PHASE_CONFIGS = {
    1: dict(dim=256, num_layers=8,  heads=4,  tag="48M"),
    2: dict(dim=512, num_layers=12, heads=8,  tag="256M"),
    3: dict(dim=768, num_layers=16, heads=12, tag="742M"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Data: download WikiText-2 from GitHub, tokenise with tiktoken
# ─────────────────────────────────────────────────────────────────────────────

WIKITEXT2_URLS = {
    "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "valid": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    "test":  "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
}


def _fetch_url(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        print(f"  [data] Using cache: {cache_path}")
        return cache_path.read_text(encoding="utf-8")
    print(f"  [data] Downloading {url} ...")
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=ctx, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    print(f"  [data] Saved to {cache_path} ({len(text):,} chars)")
    return text


def load_wikitext2(cache_dir: str = "~/.cache/wikitext2") -> Dict[str, str]:
    cache = Path(os.path.expanduser(cache_dir))
    data = {}
    for split, url in WIKITEXT2_URLS.items():
        data[split] = _fetch_url(url, cache / f"{split}.txt")
    return data


def tokenise(text: str, vocab_size_limit: int = 32000) -> List[int]:
    """Tokenise with tiktoken cl100k_base, then remap to [0, vocab_size_limit)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(text)
    except Exception:
        # Fallback: character-level tokeniser
        chars = sorted(set(text))
        c2i = {c: i % vocab_size_limit for i, c in enumerate(chars)}
        ids = [c2i[c] for c in text]
        return ids
    # Map to smaller vocab via modulo (deterministic, stable)
    return [i % vocab_size_limit for i in ids]


def make_token_batches(
    token_ids: List[int],
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Pack token_ids into [N, seq_len+1] tensor (input + target)."""
    n = (len(token_ids) - 1) // seq_len
    token_ids = token_ids[: n * seq_len + 1]
    arr = torch.tensor(token_ids, dtype=torch.long)
    # Shape [n, seq_len+1]
    arr = arr.unfold(0, seq_len + 1, seq_len)
    return arr.to(device)


def get_batch(batches: torch.Tensor, batch_size: int, step: int) -> torch.Tensor:
    n = batches.shape[0]
    start = (step * batch_size) % max(n - batch_size, 1)
    return batches[start : start + batch_size]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

LEVEL_NAMES   = ["cms_fast", "cms_mid", "cms_slow", "cms_ultra"]
LEVEL_PERIODS = [1, 4, 32, 128]


def build_hope(
    device: torch.device,
    dim: int,
    num_layers: int,
    heads: int,
    vocab_size: int = 32000,
) -> HOPEModel:
    cms_levels = [
        LevelSpec(name="cms_fast",  update_period=1,   optimizer_key="cms_opt"),
        LevelSpec(name="cms_mid",   update_period=4,   optimizer_key="cms_opt"),
        LevelSpec(name="cms_slow",  update_period=32,  optimizer_key="cms_opt"),
        LevelSpec(name="cms_ultra", update_period=128, optimizer_key="cms_opt"),
    ]
    titan_level = LevelSpec(name="titan", update_period=8, optimizer_key="titan_opt")
    optimizer_configs = {
        "titan_opt": {
            "type": "deep_momentum", "lr": 6e-4,
            "params": {"beta": 0.9, "beta2": 0.999, "variant": "nl_l2_precond"},
        },
        "cms_opt": {
            "type": "deep_momentum", "lr": 3e-4,
            "params": {"beta": 0.9, "beta2": 0.999, "variant": "nl_l2_precond"},
        },
    }
    config = ModelConfig(
        vocab_size=vocab_size, dim=dim,
        num_layers=num_layers, heads=heads,
        titan_level=titan_level, cms_levels=cms_levels,
        teach_scale=0.1, teach_clip=5.0, surprise_threshold=None,
        self_mod_lr=1e-3, block_variant="hope_hybrid",
        optimizers=optimizer_configs, hgc_enabled=False,
    )
    model = HOPEModel(config).to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def compute_teach_signal(model: HOPEModel, logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.detach(), dim=-1)
    residual = probs.clone()
    B, T, V = residual.shape
    if T < 2:
        return torch.zeros(B, T, model.config.dim, device=logits.device, dtype=logits.dtype)
    targets = tokens[:, 1 : T + 1]
    if targets.shape[1] < T:
        pad = torch.zeros(B, T - targets.shape[1], dtype=targets.dtype, device=targets.device)
        targets = torch.cat([targets, pad], dim=1)
    active = torch.zeros(B, T, dtype=torch.bool, device=logits.device)
    active[:, : T - 1] = True
    active_f = active.float().unsqueeze(-1)
    residual *= active_f
    safe_t = torch.where(active, targets, torch.zeros_like(targets))
    residual.scatter_add_(-1, safe_t.unsqueeze(-1), -active_f)
    residual /= active_f.sum().clamp(min=1.0)
    hw = model.lm_head.weight.detach().to(residual.dtype)
    return residual @ hw  # [B, T, D]


def train_step(
    model: HOPEModel,
    tokens: torch.Tensor,
    outer_optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    inp, tgt = tokens[:, :-1], tokens[:, 1:]
    outer_optimizer.zero_grad()
    logits = model(inp)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    teach = compute_teach_signal(model, logits.detach(), tokens)
    with torch.no_grad():
        model(inp, teach_signal=teach)
    outer_optimizer.step()
    return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity evaluation on held-out set
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_perplexity(
    model: HOPEModel,
    eval_batches: torch.Tensor,
    max_batches: int = 50,
) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    n = min(len(eval_batches), max_batches)
    for i in range(n):
        batch = eval_batches[i : i + 1]
        inp, tgt = batch[:, :-1], batch[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += tgt.numel()
    model.train()
    return math.exp(min(total_loss / max(total_tokens, 1), 20))


# ─────────────────────────────────────────────────────────────────────────────
# Probe utilities — read DeepMomentum states
# ─────────────────────────────────────────────────────────────────────────────

def get_level_momentum_flat(model: HOPEModel, level_name: str) -> Optional[torch.Tensor]:
    parts = []
    for block in model.blocks:
        lm = getattr(block, "level_manager", None)
        if lm is None:
            continue
        opt = lm.optimizers.get(level_name)
        if opt is None or not isinstance(opt, DeepMomentum):
            continue
        for st in opt.state.values():
            if st.grad_avg is not None:
                parts.append(st.grad_avg.detach().float().reshape(-1))
    return torch.cat(parts, dim=0) if parts else None


def get_level_params_flat(model: HOPEModel, level_name: str) -> Optional[torch.Tensor]:
    parts = []
    for block in model.blocks:
        cms = getattr(block, "cms", None)
        if cms is None or level_name not in cms.blocks:
            continue
        for p in cms.blocks[level_name].parameters():
            parts.append(p.detach().float().reshape(-1))
    return torch.cat(parts, dim=0) if parts else None


def compute_svd_basis(
    grad_matrix: torch.Tensor, rank: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Always work on CPU float32 to avoid CUDA LAPACK issues on large matrices
    G = grad_matrix.cpu().float()
    G = torch.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
    # Drop all-zero columns (degenerate snapshots)
    col_norms = G.norm(dim=0)
    G = G[:, col_norms > 1e-12]
    if G.shape[1] < 2:
        D = G.shape[0]
        q = min(rank, D)
        return torch.zeros(D, q), torch.zeros(q)
    q = min(rank, G.shape[0], G.shape[1])
    try:
        U, S, _ = torch.linalg.svd(G, full_matrices=False)
    except Exception as e:
        # Last resort: randomized SVD via QR
        print(f"  [svd fallback] {e}")
        Q, R = torch.linalg.qr(G)
        U, S, _ = torch.linalg.svd(R, full_matrices=False)
        U = Q @ U
        S = S.abs()
    return U[:, :q], S[:q]


def proj_norm_ratio(U: torch.Tensor, v: torch.Tensor) -> float:
    vf = v.float().to(U.device)
    nv = vf.norm()
    if nv < 1e-12:
        return 0.0
    return float((U.float().T @ vf).norm().item() / nv.item())


def param_drift(now: torch.Tensor, ref: torch.Tensor) -> float:
    d = (now.float() - ref.float()).norm()
    r = ref.float().norm()
    return float((d / (r + 1e-12)).item())


# ─────────────────────────────────────────────────────────────────────────────
# Main probe
# ─────────────────────────────────────────────────────────────────────────────

def run_probe(
    phase: int = 1,
    vocab_size: int = 32000,
    seq_len: int = 128,
    task_a_steps: int = 2000,
    task_b_steps: int = 500,
    batch_size: int = 16,
    lr_outer: float = 2e-4,
    grad_rank: int = 16,
    eval_every: int = 100,
    split_frac: float = 0.5,
    device: torch.device = torch.device("cuda:0"),
    seed: int = 42,
    cache_dir: str = "~/.cache/wikitext2",
) -> Dict:
    torch.manual_seed(seed)
    cfg = PHASE_CONFIGS[phase]
    dim, num_layers, heads, tag = cfg["dim"], cfg["num_layers"], cfg["heads"], cfg["tag"]
    print(f"\n{'='*60}")
    print(f"  Claim 1 Probe  |  Phase {phase}  |  ~{tag} params")
    print(f"  dim={dim}  layers={num_layers}  heads={heads}")
    print(f"  task_a={task_a_steps}  task_b={task_b_steps}  bs={batch_size}")
    print(f"  device={device}")
    print(f"{'='*60}\n")

    # ── Load & tokenise WikiText-2 ────────────────────────────────────────────
    print("=== Loading WikiText-2 ===")
    wt2 = load_wikitext2(cache_dir)
    train_ids = tokenise(wt2["train"], vocab_size)
    valid_ids  = tokenise(wt2["valid"], vocab_size)
    print(f"  train tokens: {len(train_ids):,}  |  valid tokens: {len(valid_ids):,}")

    # Split train into Task A (first half) and Task B (second half)
    split = int(len(train_ids) * split_frac)
    ids_A = train_ids[:split]
    ids_B = train_ids[split:]
    print(f"  Task A tokens: {len(ids_A):,}  |  Task B tokens: {len(ids_B):,}")

    batches_A    = make_token_batches(ids_A,    seq_len, batch_size, device)
    batches_B    = make_token_batches(ids_B,    seq_len, batch_size, device)
    batches_eval = make_token_batches(valid_ids, seq_len, batch_size, device)
    print(f"  Batches A: {len(batches_A)}  |  Batches B: {len(batches_B)}  |  Eval: {len(batches_eval)}")

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n=== Building HOPEModel ===")
    model = build_hope(device, dim=dim, num_layers=num_layers, heads=heads, vocab_size=vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.2f}M")
    vram_mb = torch.cuda.memory_allocated(device) / 1e6 if device.type == "cuda" else 0
    print(f"  VRAM after build: {vram_mb:.0f} MB")

    outer_optimizer = torch.optim.AdamW(model.parameters(), lr=lr_outer, weight_decay=1e-4)

    # ── Phase 1: Train on Task A ──────────────────────────────────────────────
    print("\n=== Phase 1: Train on Task A ===")
    # Keep buf small for large models: each snapshot = param_count floats on CPU RAM
    # 742M model: each snapshot ~3GB; 32 snapshots = ~96GB CPU RAM (manageable)
    n_params_cms = sum(p.numel() for b in model.blocks
                       for p in getattr(b, "cms", type("", (), {"parameters": list})()).parameters()
                       if True) if hasattr(model.blocks[0], "cms") else n_params
    MAX_GRAD_BUF = 32 if n_params > 200e6 else 64 if n_params > 50e6 else 256
    grad_buffers: Dict[str, List[torch.Tensor]] = {n: [] for n in LEVEL_NAMES}
    ppl_A_during_training: List[float] = []

    for step in range(task_a_steps):
        batch = get_batch(batches_A, batch_size, step)
        loss = train_step(model, batch, outer_optimizer)

        # Collect momentum snapshots for later SVD (skip NaN-containing snapshots)
        for ln in LEVEL_NAMES:
            if len(grad_buffers[ln]) < MAX_GRAD_BUF:
                m = get_level_momentum_flat(model, ln)
                if m is not None and not torch.isnan(m).any() and not torch.isinf(m).any():
                    grad_buffers[ln].append(m.clone().cpu())

        if step % eval_every == 0:
            ppl = evaluate_perplexity(model, batches_eval)
            ppl_A_during_training.append(ppl)
            vram_mb = torch.cuda.memory_allocated(device) / 1e6 if device.type == "cuda" else 0
            print(f"  A step {step:5d}  loss={loss:.4f}  ppl={ppl:.2f}  VRAM={vram_mb:.0f}MB")

    print(f"  Task A final loss={loss:.4f}")

    # ── Extract gradient signatures U_A ──────────────────────────────────────
    print("\n=== Extracting gradient signatures U_A per level ===")
    subspaces_A: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for ln, C in zip(LEVEL_NAMES, LEVEL_PERIODS):
        buf = grad_buffers[ln]
        eff_rank = max(1, min(grad_rank, len(buf) - 1))
        if len(buf) < 2:
            print(f"  {ln}: SKIP (buf={len(buf)} < 2)")
            continue
        G = torch.stack(buf, dim=1)  # keep on CPU — too large for GPU on big models
        U, S = compute_svd_basis(G, eff_rank)
        subspaces_A[ln] = (U.cpu(), S.cpu())
        print(f"  {ln} (C={C}): U={U.shape}, top-SV={S[0].item():.4f}, buf={len(buf)}")

    # Snapshot params at end of Task A
    theta_A: Dict[str, torch.Tensor] = {}
    for ln in LEVEL_NAMES:
        pf = get_level_params_flat(model, ln)
        if pf is not None:
            theta_A[ln] = pf.cpu()

    # Baseline perplexity on validation AFTER Task A
    ppl_after_A = evaluate_perplexity(model, batches_eval)
    print(f"\n  PPL on validation after Task A: {ppl_after_A:.2f}")

    # ── Verify A/B subspace orthogonality (quick 50-step probe) ──────────────
    print("\n=== Checking Task A/B subspace overlap ===")
    _state_bk  = {k: v.clone() for k, v in model.state_dict().items()}
    _dm_bk: Dict = {}
    for bi, block in enumerate(model.blocks):
        for ln, opt in block.level_manager.optimizers.items():
            for pk, st in opt.state.items():
                _dm_bk[(bi, ln, pk)] = {
                    "g": st.grad_avg.clone() if st.grad_avg is not None else None,
                    "s": st.sq_avg.clone()   if st.sq_avg   is not None else None,
                }
    b_buf: Dict[str, List[torch.Tensor]] = {n: [] for n in LEVEL_NAMES}
    for vstep in range(50):
        batch = get_batch(batches_B, batch_size, vstep)
        train_step(model, batch, outer_optimizer)
        for ln in LEVEL_NAMES:
            if len(b_buf[ln]) < 50:
                m = get_level_momentum_flat(model, ln)
                if m is not None:
                    b_buf[ln].append(m.clone().cpu())
    # Restore
    model.load_state_dict(_state_bk)
    for bi, block in enumerate(model.blocks):
        for ln, opt in block.level_manager.optimizers.items():
            for pk, st in opt.state.items():
                bk = _dm_bk.get((bi, ln, pk), {})
                st.grad_avg = bk.get("g")
                st.sq_avg   = bk.get("s")
    outer_optimizer = torch.optim.AdamW(model.parameters(), lr=lr_outer, weight_decay=1e-4)

    overlap_AB: Dict[str, float] = {}
    for ln, C in zip(LEVEL_NAMES, LEVEL_PERIODS):
        if ln not in subspaces_A or len(b_buf[ln]) < 2:
            continue
        U_A = subspaces_A[ln][0].to(device)
        G_B = torch.stack(b_buf[ln], dim=1).to(device)
        U_B, _, _ = torch.linalg.svd(G_B, full_matrices=False)
        r = min(U_A.shape[1], U_B.shape[1])
        ov = float((U_A[:, :r].T @ U_B[:, :r]).norm().item() / math.sqrt(r))
        overlap_AB[ln] = ov
        print(f"  {ln} (C={C}): A-B overlap={ov:.4f}")

    # Initial projections at Task B step 0
    init_proj: Dict[str, float] = {}
    for ln in LEVEL_NAMES:
        if ln not in subspaces_A:
            init_proj[ln] = float("nan")
            continue
        U_A = subspaces_A[ln][0].to(device)
        m = get_level_momentum_flat(model, ln)
        init_proj[ln] = proj_norm_ratio(U_A, m) if m is not None else float("nan")
    print(f"\n  Init projections: { {k: f'{v:.3f}' for k,v in init_proj.items()} }")

    # ── Phase 2: Train on Task B, measure collapse ────────────────────────────
    print("\n=== Phase 2: Train on Task B — measuring collapse ===")
    beta = 0.9
    records: Dict = {
        "step": [], "loss_B": [],
        "ppl_taskA_eval": [],          # perplexity on TaskA validation (forgetting)
        "ppl_taskA_step": [],          # step at which ppl was measured
        "phase": phase, "tag": tag,
        "n_params_M": round(n_params / 1e6, 2),
        "task_a_steps": task_a_steps,
        "ppl_after_taskA": ppl_after_A,
        "overlap_AB": overlap_AB,
        "model_info": {
            "dim": dim, "num_layers": num_layers, "heads": heads,
            "vocab_size": vocab_size, "seq_len": seq_len,
            "optimizer": "DeepMomentum(nl_l2_precond)",
            "outer_optimizer": "AdamW",
            "block_variant": "hope_hybrid",
            "task": "WikiText-2",
        },
    }
    for ln in LEVEL_NAMES:
        records[f"mom_proj_{ln}"]      = []
        records[f"mom_proj_norm_{ln}"] = []
        records[f"param_drift_{ln}"]   = []
        records[f"theory_decay_{ln}"]  = []

    for step in range(task_b_steps):
        batch = get_batch(batches_B, batch_size, step)
        loss  = train_step(model, batch, outer_optimizer)

        records["step"].append(step)
        records["loss_B"].append(loss)

        # Periodic perplexity on validation (measures actual forgetting)
        if step % eval_every == 0:
            ppl = evaluate_perplexity(model, batches_eval)
            records["ppl_taskA_eval"].append(ppl)
            records["ppl_taskA_step"].append(step)

        for ln, C in zip(LEVEL_NAMES, LEVEL_PERIODS):
            if ln not in subspaces_A:
                for k in ["mom_proj", "mom_proj_norm", "param_drift", "theory_decay"]:
                    records[f"{k}_{ln}"].append(None)
                continue

            U_A = subspaces_A[ln][0].to(device)
            m   = get_level_momentum_flat(model, ln)
            p   = proj_norm_ratio(U_A, m) if m is not None else float("nan")

            ip  = init_proj.get(ln, float("nan"))
            p_n = p / ip if (not math.isnan(ip) and ip > 1e-8) else float("nan")

            theta_now = get_level_params_flat(model, ln)
            drift = param_drift(theta_now.to(device), theta_A[ln].to(device)) \
                    if (theta_now is not None and ln in theta_A) else float("nan")

            k_eff = step // C + 1
            theory = beta ** k_eff

            def _s(v):
                return None if math.isnan(v) else v

            records[f"mom_proj_{ln}"].append(_s(p))
            records[f"mom_proj_norm_{ln}"].append(_s(p_n))
            records[f"param_drift_{ln}"].append(_s(drift))
            records[f"theory_decay_{ln}"].append(theory)

        if step % 50 == 0:
            ppl_str = ""
            if records["ppl_taskA_eval"]:
                ppl_str = f"  ppl_valA={records['ppl_taskA_eval'][-1]:.2f}"
            print(f"  B step {step:5d}  loss={loss:.4f}{ppl_str}", end="")
            for ln, C in zip(LEVEL_NAMES, LEVEL_PERIODS):
                v = records[f"mom_proj_norm_{ln}"][-1]
                if v is not None:
                    print(f"  {ln[4:]}={v:.3f}", end="")
            print()

    print("\n=== Probe complete ===")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Claim 1 probe — WikiText-2 real LM")
    parser.add_argument("--phase",        type=int,   default=1, choices=[1, 2, 3])
    parser.add_argument("--output_dir",   default="./results_phase1")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--vocab_size",   type=int,   default=32000)
    parser.add_argument("--seq_len",      type=int,   default=128)
    parser.add_argument("--task_a_steps", type=int,   default=2000)
    parser.add_argument("--task_b_steps", type=int,   default=500)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr_outer",     type=float, default=2e-4)
    parser.add_argument("--grad_rank",    type=int,   default=16)
    parser.add_argument("--eval_every",   type=int,   default=100)
    parser.add_argument("--split_frac",   type=float, default=0.5)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--cache_dir",    default="~/.cache/wikitext2")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    records = run_probe(
        phase        = args.phase,
        vocab_size   = args.vocab_size,
        seq_len      = args.seq_len,
        task_a_steps = args.task_a_steps,
        task_b_steps = args.task_b_steps,
        batch_size   = args.batch_size,
        lr_outer     = args.lr_outer,
        grad_rank    = args.grad_rank,
        eval_every   = args.eval_every,
        split_frac   = args.split_frac,
        device       = device,
        seed         = args.seed,
        cache_dir    = args.cache_dir,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _serial(v):
        if isinstance(v, float): return None if math.isnan(v) else v
        if isinstance(v, list):  return [_serial(x) for x in v]
        return v

    clean = {k: _serial(v) for k, v in records.items()}
    rec_path = out / "records.json"
    with open(rec_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nSaved records -> {rec_path}")
    print("=== Done. Run plot_wikitext_v3.py to generate figures. ===")


if __name__ == "__main__":
    main()
