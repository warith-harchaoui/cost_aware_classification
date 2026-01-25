# Cost-Aware Classification + Fraud Benchmark (IEEE-CIS)

**Author:** Warith Harchaoui <wharchaoui@nexton-group.com>

Corporate research/engineering repository for **cost-aware classification** with **example-dependent misclassification costs**.

## Overview

This repository implements multiple loss functions for cost-aware classification, where different misclassifications have different costs. Traditional cross-entropy treats all errors equally, but in real-world scenarios (e.g., fraud detection), some mistakes are more expensive than others.

## 📋 Available Loss Functions

### Baseline Losses

#### 1. **Cross-Entropy** (`cross_entropy`)
Standard cross-entropy loss without cost awareness.

**When to use:** Baseline comparison when all misclassifications have equal cost.

**Run command:**
```bash
python -m examples.fraud_detection --loss cross_entropy --epochs 5 --run-id baseline
```

#### 2. **Weighted Cross-Entropy** (`cross_entropy_weighted`)
Sample-weighted cross-entropy with weights $w_i = C_i[y_i, 1-y_i]$ derived from the cost matrix.

**When to use:** Simple cost-aware baseline that reweights examples by their misclassification cost.

**Run command:**
```bash
python -m examples.fraud_detection --loss cross_entropy_weighted --epochs 5 --run-id weighted_baseline
```

**Additional parameters:**
- Automatically uses cost matrix to compute sample weights
- Weights are normalized by median for stability

### Cost-Aware Losses (Optimal Transport)

All OT-based losses use a cost matrix $C$ where $C_{ij}$ represents the cost of predicting class $j$ when the true class is $i$.

#### 3. **Sinkhorn-Fenchel-Young Loss** (`sinkhorn_fenchel_young`)
Implicit Fenchel–Young loss with Frank–Wolfe inner solver. Does not differentiate through the inner optimization.

**Theory:**
- Uses envelope theorem: computes optimal solution without backpropagating through solver
- Frank-Wolfe algorithm solves inner QP on the simplex
- Stable gradients, efficient computation

**When to use:** When you want provably stable gradients from implicit differentiation.

**Run command:**
```bash
python -m examples.fraud_detection --loss sinkhorn_fenchel_young --epochs 5 --run-id fenchel_young
```

**Advanced options:**
```bash
# Custom epsilon (regularization parameter)
python -m examples.fraud_detection --loss sinkhorn_fenchel_young --epochs 5 \
  --epsilon 0.5 --run-id fy_eps05

# More inner solver iterations for better convergence
python -m examples.fraud_detection --loss sinkhorn_fenchel_young --epochs 5 \
  --cacis-solver-iter 100 --run-id fy_iter100
```

**Hyperparameters:**
- `--epsilon`: Regularization parameter ε (default: auto-computed from cost matrix)
- `--cacis-solver-iter`: Frank-Wolfe iterations (default: 50)

#### 4. **Sinkhorn Envelope Loss** (`sinkhorn_envelope`)
Entropic OT loss with custom Sinkhorn solver and envelope-style gradients.

**Theory:**
- Solves entropic OT with custom Sinkhorn iterations
- Envelope gradient: treats optimal transport plan as constant during backprop
- Keeps explicit dependence on predictions through KL term
- Memory efficient, stable gradients

**When to use:** When you want full control over the Sinkhorn implementation with stable gradients.

**Run command:**
```bash
python -m examples.fraud_detection --loss sinkhorn_envelope --epochs 5 --run-id envelope
```

**Advanced options:**
```bash
# More Sinkhorn iterations for convergence
python -m examples.fraud_detection --loss sinkhorn_envelope --epochs 5 \
  --sinkhorn-max-iter 100 --run-id envelope_iter100

# Fixed epsilon value
python -m examples.fraud_detection --loss sinkhorn_envelope --epochs 5 \
  --epsilon 0.1 --run-id envelope_eps01
```

**Hyperparameters:**
- `--sinkhorn-max-iter`: Sinkhorn iterations (default: 50)
- `--epsilon`: Regularization ε (default: auto from cost matrix)
- `--epsilon-mode`: How to compute ε: `constant`, `offdiag_mean`, `offdiag_median`, `offdiag_max`

#### 5. **Sinkhorn Full Autodiff Loss** (`sinkhorn_autodiff`)
Entropic OT loss with full differentiation through Sinkhorn iterations.

**Theory:**
- Backpropagates through all Sinkhorn iterations
- More "end-to-end" but higher memory usage
- May have less stable gradients for many iterations

**When to use:** When you want to compare end-to-end learning vs envelope gradients, or when iteration count is small.

**Run command:**
```bash
python -m examples.fraud_detection --loss sinkhorn_autodiff --epochs 5 --run-id autodiff
```

**Advanced options:**
```bash
# Fewer iterations to reduce memory (important for autodiff!)
python -m examples.fraud_detection --loss sinkhorn_autodiff --epochs 5 \
  --sinkhorn-max-iter 30 --run-id autodiff_iter30
```

**Hyperparameters:**
- `--sinkhorn-max-iter`: Sinkhorn iterations (default: 50, consider reducing for memory)
- `--epsilon`: Regularization ε

**Memory considerations:** This loss stores intermediate tensors for backprop. Use fewer iterations if memory is limited.

#### 6. **Sinkhorn POT Loss** (`sinkhorn_pot`)
Entropic OT loss using the [Python Optimal Transport (POT)](https://pythonot.github.io/) library with envelope gradients.

**Theory:**
- Uses POT's battle-tested, optimized Sinkhorn implementation
- Envelope gradients (same as `sinkhorn_envelope`) for stability
- GPU acceleration via POT's backend
- Best numerical stability from mature implementation

**When to use:** Production scenarios requiring reliability, or when comparing custom implementations against established library.

**Run command:**
```bash
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 --run-id pot
```

**Advanced options:**
```bash
# More iterations for convergence
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --sinkhorn-max-iter 100 --run-id pot_iter100

# Custom epsilon
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --epsilon 0.2 --run-id pot_eps02
```

**Hyperparameters:**
- `--sinkhorn-max-iter`: Maximum Sinkhorn iterations (default: 50)
- `--epsilon`: Regularization ε (default: auto from cost matrix)

**Benefits:**
- Mature, optimized implementation
- GPU support through POT backends
- Numerical stability improvements
- Active maintenance by POT community

## 🚀 Complete Usage Guide

### Installation

Install conda if you haven't already. See [conda](https://harchaoui.org/warith/4ml) for instructions.

```bash
PROJECT=cost_aware_fraud
GITHUB_OWNER=warith-harchaoui

REPOURL=https://github.com/$GITHUB_OWNER/$PROJECT.git
ENV=env4$PROJECT

git clone $REPOURL
cd $PROJECT
conda update -y -n base -c defaults conda
conda create -y -n $ENV python=3.10
conda activate $ENV
conda install -y pip
pip install -r requirements.txt
```

### Dataset Download

Download the IEEE-CIS fraud detection dataset:

```bash
rm -rf ieee-fraud-detection.zip ieee-fraud-detection || true
mkdir ieee-fraud-detection
wget -c http://deraison.ai/ai/ieee-fraud-detection.zip
unzip ieee-fraud-detection.zip -d ieee-fraud-detection
```

### Basic Usage Examples

#### Run a Single Loss
```bash
# Train with cross-entropy for 5 epochs
python -m examples.fraud_detection --loss cross_entropy --epochs 5 --run-id exp1

# Train with SinkhornPOTLoss
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 10 --run-id pot_exp

# Train with Fenchel-Young loss
python -m examples.fraud_detection --loss sinkhorn_fenchel_young --epochs 5 --run-id fy_exp
```

#### Run All Losses (Benchmark)
```bash
# Compare all 6 loss functions
python -m examples.fraud_detection --loss all --epochs 5 --run-id benchmark1
```

This will train 6 separate models (one per loss) and save results in:
```
fraud_output/benchmark1/cross_entropy/
fraud_output/benchmark1/cross_entropy_weighted/
fraud_output/benchmark1/sinkhorn_fenchel_young/
fraud_output/benchmark1/sinkhorn_envelope/
fraud_output/benchmark1/sinkhorn_autodiff/
fraud_output/benchmark1/sinkhorn_pot/
```

### Advanced Options

#### Resume Training
Continue training from a checkpoint for additional epochs:

```bash
# Train 5 epochs initially
python -m examples.fraud_detection --loss all --epochs 5 --run-id longrun

# Continue for 3 MORE epochs (total: 8)
python -m examples.fraud_detection --loss all --epochs 3 --run-id longrun --resume
```

#### Model Architecture
```bash
# Use linear model instead of MLP
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --backbone linear --run-id linear_model

# Custom MLP architecture
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --backbone mlp --hidden-dims 512,256,128 --dropout 0.2 --run-id deep_mlp

# Disable batch normalization
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --no-batchnorm --run-id no_bn
```

#### Training Configuration
```bash
# Custom batch size and learning rate
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 10 \
  --batch-size 1024 --lr 5e-4 --run-id custom_training

# Quick test run (few batches per epoch)
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 2 \
  --quick --run-id quick_test

# Custom train/val split
python -m examples.fraud_detection --loss all --epochs 5 \
  --split 0.2 --run-id split20
```

#### Device Selection
```bash
# Force CPU
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --device cpu --run-id cpu_run

# Force CUDA (if available)
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --device cuda --run-id gpu_run

# Use MPS (Apple Silicon)
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --device mps --run-id mps_run

# Auto-detect (default)
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --device auto --run-id auto_device
```

#### Loss-Specific Hyperparameters
```bash
# Sinkhorn variants: control iterations
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --sinkhorn-max-iter 200 --run-id pot_iter200

# Fenchel-Young: control Frank-Wolfe iterations
python -m examples.fraud_detection --loss sinkhorn_fenchel_young --epochs 5 \
  --cacis-solver-iter 200 --run-id fy_iter200

# All Sinkhorn: fixed regularization
python -m examples.fraud_detection --loss sinkhorn_envelope --epochs 5 \
  --epsilon 0.5 --run-id env_eps05
```

### Output Artifacts

Each run creates a directory `fraud_output/<run-id>/<loss_name>/` with:

```
checkpoint_last.pt          # Latest checkpoint (for resuming)
checkpoint_best.pt          # Best checkpoint (by PR-AUC)
train_ema_metrics.csv       # Training metrics (EMA smoothed)
probe_metrics.csv           # Validation metrics over time
precision_recall_curve.png  # PR curve visualization
probe_pr_auc.png           # PR-AUC vs iteration
probe_expected_opt_regret.png      # Expected optimal regret
probe_realized_regret.png          # Realized regret
train_expected_opt_regret_ema.png  # Training regret (EMA)
train_realized_regret_ema.png      # Training realized regret
```

### Metrics Explained

- **PR-AUC (Precision-Recall Area Under Curve):** Primary metric for imbalanced fraud detection
- **Expected Optimal Regret:** Expected cost under optimal decision-making given predictions
- **Realized Regret:** Actual cost incurred from model predictions
- Lower regret = better cost-aware performance

## 🎯 Choosing a Loss Function

| Loss | Pros | Cons | Best For |
|------|------|------|----------|
| `cross_entropy` | Simple, fast, well-understood | Ignores costs | Baseline comparison |
| `cross_entropy_weighted` | Simple cost integration | Doesn't optimize transport | Quick cost-aware baseline |
| `sinkhorn_fenchel_young` | Provably stable gradients | Custom implementation | Research, reproducibility |
| `sinkhorn_envelope` | Stable, memory efficient | Custom implementation | When memory is limited |
| `sinkhorn_autodiff` | End-to-end learning | High memory, less stable | Research comparison |
| `sinkhorn_pot` | Production-ready, optimized | External dependency | Production deployments |

**Recommendation for production:** Start with `sinkhorn_pot` for reliability and performance.

**Recommendation for research:** Compare `sinkhorn_envelope`, `sinkhorn_autodiff`, and `sinkhorn_fenchel_young` to understand gradient quality vs computational trade-offs.

## 📚 Documentation

- [`docs/math.md`](docs/math.md) — Mathematical derivations and the explicit mapping between $\varepsilon$ and POT's `reg` parameter
- [`docs/fraud_business_and_cost_matrix.md`](docs/fraud_business_and_cost_matrix.md) — Business value model and per-example cost matrix construction for fraud detection
- [`examples/sinkhorn_pot_example.py`](examples/sinkhorn_pot_example.py) — Standalone example demonstrating SinkhornPOTLoss usage

## 🔬 Example: Comparing All Losses

```bash
# Comprehensive benchmark with all losses
python -m examples.fraud_detection \
  --loss all \
  --epochs 10 \
  --batch-size 512 \
  --lr 1e-4 \
  --sinkhorn-max-iter 100 \
  --cacis-solver-iter 100 \
  --run-id comprehensive_benchmark

# Results will be in fraud_output/comprehensive_benchmark/<loss_name>/
```

Compare results by examining:
1. **PR-AUC**: Which loss achieves highest precision-recall?
2. **Regret metrics**: Which loss minimizes business costs?
3. **Training curves**: Which loss converges fastest/most stably?
4. **Computation time**: Which loss is most efficient?

## 📜 License

**Unlicense** — This is free and unencumbered software released into the public domain.  
See [UNLICENSE](https://unlicense.org) for details.

## 🙏 Acknowledgments

- POT library: https://pythonot.github.io/
- IEEE-CIS Kaggle competition for the dataset
