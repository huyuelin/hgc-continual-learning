# LAOSP: Level-Aware Optimizer-State Protection for Continual Learning in Nested Architectures

This repository contains the official implementation of the paper:

> **Level-Aware Optimizer-State Protection for Continual Learning in Nested Architectures**

## Overview

Gradient subspace projection has become a standard technique for continual learning, but existing methods operate at the parameter level, leaving the optimizer's internal state unprotected. In momentum-based optimizers (Adam, SGD with momentum), gradient memory resides in the momentum buffer, which decays old-task information regardless of parameter-level protection.

This gap becomes critical in Nested Learning (NL) architectures such as HOPE, where components update at different frequencies. We show that momentum buffers at every frequency level decay at the same rate, independent of update frequency -- a failure mode invisible to parameter-level methods.

**LAOSP** applies orthogonal gradient projection directly to the momentum buffer with protection strength scaled to each level's update frequency.

### Key Results

- Optimizer-state projection reduces forgetting by **18%** in sequential evaluation
- **23--31%** reduction in PPL-based forgetting at 150M--300M scale
- Parameter-level methods (EWC, OGD) provide no benefit or actively interfere
- Where subspace projection is applied matters as much as the projection itself

## Repository Structure

```
.
├── models/
│   ├── nested_learning/          # Base Nested Learning / HOPE implementation
│   │   ├── train.py              # Single-GPU training
│   │   ├── train_fsdp.py         # FSDP distributed training
│   │   ├── train_deepspeed.py    # DeepSpeed training
│   │   └── scripts/eval/         # Evaluation scripts
│   └── nested_learning_hgc/      # HGC-modified training with LAOSP
│       ├── train.py              # Training with OGP + CAM + CLGD
│       ├── train_fsdp.py
│       └── train_deepspeed.py
├── experiments/
│   ├── claim1_gradient_memory_collapse/  # Gradient memory decay analysis
│   ├── claim2_ogp_cam/                  # OGP/CAM ablation studies
│   ├── probe_replay_baseline.py         # Replay baseline comparison
│   ├── probe_oaks_online_triggers.py    # Online trigger analysis
│   ├── probe_realworld_seq.py           # Real-world sequential evaluation
│   └── probe_ckl_knowledge_evaluation.py # Knowledge retention evaluation
├── scripts/
│   ├── tables/                   # LaTeX table generation
│   └── plotting/                 # Figure generation
├── figures/                      # Generated figures
└── docs/                         # Documentation
```

## Method

LAOSP consists of three mechanisms:

1. **Orthogonal Gradient Projection (OGP)**: Applies subspace projection at the optimizer-state level rather than the parameter level. Before each momentum update, the incoming gradient is projected orthogonal to the old-task gradient subspace captured via SVD.

2. **Consolidation-Aware Momentum (CAM)**: Counteracts the natural EMA decay of old-task components in the momentum buffer. For longer training horizons, CAM periodically re-injects the protected subspace direction to prevent exponential decay from erasing gradient memory.

3. **Cross-Level Gradient Distillation (CLGD)**: Transfers gradient statistics between adjacent frequency levels in the CMS hierarchy. When task data is asymmetrically distributed across levels, CLGD enables knowledge flow from well-informed levels to those with limited gradient signal.

The protection strength follows a logarithmic schedule:

```
alpha(l) = 1 - 1 / log2(C(l) + 1)
```

where `C(l)` is the update period at level `l`. This assigns stronger protection to slower (more persistent) levels and weaker protection to faster (more volatile) levels.

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- CUDA >= 11.8

## Quick Start

### Training with LAOSP

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python models/nested_learning_hgc/train.py \
    --condition full_hgc \
    --device cuda:0

# Distributed (FSDP)
torchrun --nproc_per_node=4 models/nested_learning_hgc/train_fsdp.py \
    --condition full_hgc
```

### Running Experiments

```bash
# Gradient memory collapse analysis (Claim 1)
python experiments/claim1_gradient_memory_collapse/probe_real_hope.py

# OGP/CAM ablation (Claim 2)
python experiments/claim2_ogp_cam/probe_claim2_ogp_cam.py

# Sequential domain evaluation
python experiments/probe_realworld_seq.py
```

### Generating Tables and Figures

```bash
python scripts/tables/collect_and_generate_tables.py
python scripts/plotting/plot_hgc_figures.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{laosp2026,
  title={Level-Aware Optimizer-State Protection for Continual Learning in Nested Architectures},
  author={Anonymous},
  year={2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This work builds on the Nested Learning framework and HOPE architecture by Behrouz et al. (2025).
