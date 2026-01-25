# Mathematics: Cost-Aware Losses

**Author:** Warith Harchaoui <wharchaoui@nexton-group.com>

This document provides mathematical foundations for the cost-aware classification losses implemented in this repository.

## Overview

This repository implements **four** cost-aware losses for multi-class classification:

1. **SinkhornFenchelYoungLoss** — Implicit Fenchel–Young with Frank–Wolfe inner solver
2. **SinkhornEnvelopeLoss** — Custom Sinkhorn with envelope gradients
3. **SinkhornFullAutodiffLoss** — Custom Sinkhorn with full autodiff
4. **SinkhornPOTLoss** — POT library Sinkhorn with envelope gradients

All losses share common ingredients:

- **Score vector** (logits): $f \in \mathbb{R}^K$
- **True label**: $y \in \{0,\dots,K-1\}$
- **Cost matrix**: $C \in \mathbb{R}_+^{K\times K}$ where $C_{i,j}$ is the business cost of predicting class $j$ when the true class is $i$
- **Assumption**: $C_{i,i}=0$ (no cost for correct predictions) and $C_{i,j}\ge 0$

### Example-Dependent Costs

All losses support **example-dependent costs**:
- $C$ can be global $(K,K)$ shared across batch, or
- $C$ can be batched $(B,K,K)$ with different costs per example

This flexibility enables modeling scenarios where misclassification costs vary per instance (e.g., fraud detection where costs depend on transaction amount).

---

## Entropic Regularization: $\varepsilon$ Parameter

### Notation Mapping: $\varepsilon$ vs POT's `reg`

Balanced entropic optimal transport between distributions $p, q \in \Delta_K$ is defined as:

```math
\mathrm{OT}_\varepsilon(p,q) \;=\; \min_{P \in \mathbb{R}_+^{K\times K}}
\;\langle P, C \rangle \;+\; \varepsilon\;\mathrm{KL}(P \,\|\, p\otimes q)
\quad\text{s.t.}\quad
P\mathbf{1} = p,\;\; P^\top \mathbf{1} = q.
```

The entropic regularization parameter is **the scalar multiplying the KL term**:

- In academic papers: typically denoted $\varepsilon$ or $\lambda$
- In **POT library**: named `reg`
- In **this codebase**: called `epsilon`

**For balanced KL-regularized formulation:**

```math
\boxed{\texttt{POT.reg} = \varepsilon = \texttt{our epsilon}}
```

This means an "apples-to-apples" comparison across implementations requires matching this scalar value.

### Adaptive Epsilon Computation (Recommended)

**Instead of manually choosing $\varepsilon$, we compute it adaptively from the cost matrix using rule-of-thumb heuristics:**

All cost-aware losses support `epsilon_mode` parameter with three adaptive methods:

1. **`offdiag_mean`** (default)
   ```math
   \varepsilon = \alpha \cdot \text{mean}(C_{i,j} : i \neq j)
   ```
   - Uses average off-diagonal cost
   - Balanced, works well in most cases
   - **Recommended as default**

2. **`offdiag_median`**
   ```math
   \varepsilon = \alpha \cdot \text{median}(C_{i,j} : i \neq j)
   ```
   - Uses median off-diagonal cost
   - **Robust to outlier costs**
   - Recommended when cost distribution has extreme values

3. **`offdiag_max`**
   ```math
   \varepsilon = \alpha \cdot \max(C_{i,j} : i \neq j)
   ```
   - Uses maximum off-diagonal cost
   - **Conservative regularization**
   - Ensures all costs are well-regularized

where $\alpha$ is the `epsilon_scale` parameter (default: 1.0).

### Epsilon Scaling

The `epsilon_scale` parameter provides fine-grained control:

- **Scale < 1.0**: Tighter regularization → sharper solutions
- **Scale = 1.0**: Default balanced setting
- **Scale > 1.0**: Looser regularization → smoother solutions

**Example:**
```python
# Adaptive epsilon with median (robust to outliers)
loss = SinkhornPOTLoss(epsilon_mode="offdiag_median", epsilon_scale=1.0)

# Tighter regularization: scale by 0.5
loss = SinkhornPOTLoss(epsilon_mode="offdiag_mean", epsilon_scale=0.5)

# Looser regularization: scale by 2.0
loss = SinkhornPOTLoss(epsilon_mode="offdiag_mean", epsilon_scale=2.0)
```

### Constant Epsilon (Advanced)

For controlled experiments, you can override with constant:
```python
# Fixed epsilon (disables adaptive computation)
loss = SinkhornPOTLoss(epsilon_mode="constant", epsilon=0.1)
```

**Use constant epsilon sparingly** — adaptive methods are more robust and require no tuning.

---

## Sinkhorn Algorithm (Balanced OT)

The Gibbs kernel is defined as:

```math
K \;=\; \exp(-C/\varepsilon)
```

Sinkhorn iterations solve for scaling vectors $u, v$ such that:

```math
P^\star = \mathrm{diag}(u)\,K\,\mathrm{diag}(v)
```

has marginals $p$ and $q$. The standard updates are:

```math
u \leftarrow \frac{p}{K v},\qquad
v \leftarrow \frac{q}{K^\top u}
```

with elementwise division and clamping for numerical stability.

**Stopping criteria:**
- Maximum iterations (`max_iter` or `sinkhorn_max_iter`)
- Convergence threshold (POT's `stopThr`, typically $10^{-9}$)

---

## Loss Function Specifications

### 1. SinkhornEnvelopeLoss

**Custom implementation with envelope-style gradients.**

#### Forward Pass

Build distributions:
- $p = \mathrm{softmax}(f)$ — predicted distribution from logits
- $q = \mathrm{onehot}(y)$ — target distribution (with label smoothing for stability)

Compute Sinkhorn plan $P^\star(p,q)$ via custom iterations, then evaluate the **primal objective**:

```math
L(p,q) = \langle P^\star, C \rangle + \varepsilon\,\mathrm{KL}(P^\star \,\|\, p\otimes q)
```

where:
```math
\mathrm{KL}(P \,\|\, p\otimes q) = \sum_{i,j} P_{i,j} \left[\log P_{i,j} - \log p_i - \log q_j\right]
```

#### Backward Pass (Envelope / Implicit Gradient)

Treat $P^\star$ as **constant** in backward pass:
- No differentiation through Sinkhorn iterations (saves memory)
- Gradients flow through $\log p$ (and $\log q$ if needed) in KL term
- Stable, memory-efficient

This matches the "don't backprop through the solver" philosophy of Sinkhorn-Fenchel-Young.

**Hyperparameters:**
- `epsilon_mode`: Adaptive method (default: `offdiag_mean`)
- `epsilon_scale`: Scale factor (default: 1.0)
- `max_iter`: Sinkhorn iterations (default: 50)
- `label_smoothing`: Stability parameter (default: 1e-3)

---

### 2. SinkhornFullAutodiffLoss

**Custom implementation with full differentiation through iterations.**

Same objective as `SinkhornEnvelopeLoss`, but **differentiates through all Sinkhorn iterations**.

**Trade-offs:**
- ✅ More "end-to-end" learning
- ❌ Higher memory usage (stores intermediate tensors)
- ❌ Potentially less stable gradients for many iterations

**When to use:** Research comparisons or when iteration count is small (<30).

**Hyperparameters:**
- `epsilon_mode`, `epsilon_scale`: Same as envelope
- `max_iter`: Consider reducing (e.g., 30) to save memory

---

### 3. SinkhornPOTLoss

**Production-ready implementation using the POT library.**

Uses POT's `ot.sinkhorn` function with envelope gradients (same gradient strategy as `SinkhornEnvelopeLoss`).

**Advantages:**
- ✅ Battle-tested, mature implementation
- ✅ Optimized for GPU via POT backends
- ✅ Superior numerical stability
- ✅ Active maintenance and community support

**Hyperparameters:**
- `epsilon_mode`, `epsilon_scale`: Adaptive epsilon control
- `max_iter`: POT Sinkhorn iterations (default: 50)
- `stopThr`: Convergence threshold (default: 1e-9)
- `method`: `"sinkhorn"` or `"sinkhorn_log"` (log-domain for small $\varepsilon$)
- `allow_numpy_fallback`: Enable CPU fallback if GPU fails

**Recommended for production deployments.**

---

### 4. SinkhornFenchelYoungLoss

**Implicit Fenchel–Young loss with Frank–Wolfe solver.**

Implements a Fenchel–Young loss of the form:

```math
\ell(y,f) = \Omega^\star_{C,\varepsilon}(f) - f_y
```

where $\Omega^\star$ is the Fenchel conjugate of a convex regularizer $\Omega$ encoding the cost geometry.

#### Details

Computing $\Omega^\star$ involves an inner optimization over the simplex:

```math
\Omega^\star(f) = -\varepsilon \log\left(\min_{\alpha \in \Delta} \alpha^\top M \alpha\right)
```

where $M_{i,j} = \exp(-(f_i + f_j + C_{i,j})/\varepsilon)$.

**Solver:** Frank-Wolfe algorithm (iterative linear minimization on simplex)

**Gradient:** Implicit — inner solution computed under `torch.no_grad()`
- Stable training dynamics
- No expensive differentiation through inner loop
- Provably correct gradients via envelope theorem

**Hyperparameters:**
- `epsilon_mode`, `epsilon_scale`: Adaptive epsilon
- `solver_iter`: Frank-Wolfe iterations (default: 50)

---

## Comparison Summary

| Loss | Gradient | Implementation | Memory | Stability | Best For |
|------|----------|----------------|--------|-----------|----------|
| `SinkhornEnvelopeLoss` | Envelope | Custom | Low | High | Memory-constrained |
| `SinkhornFullAutodiffLoss` | Full | Custom | High | Medium | Research comparison |
| `SinkhornPOTLoss` | Envelope | POT library | Low | **Highest** | **Production** ⭐ |
| `SinkhornFenchelYoungLoss` | Implicit FY | Custom FW | Low | High | Research, theory |

---

## Practical Implementation Notes

### Numerical Stability

All implementations include:

1. **Probability clamping** in $\log(\cdot)$ operations (min value: $10^{-12}$)
2. **Label smoothing** on one-hot targets:
   ```python
   q = (1 - δ) * onehot(y) + δ/K  # δ ≈ 1e-3
   ```
3. **Division clamping** in Sinkhorn updates
4. **Log-domain computations** (POT's `sinkhorn_log`) for very small $\varepsilon$

### Epsilon Guidelines

| Scenario | Recommendation |
|----------|----------------|
| **Default/Unknown** | `epsilon_mode="offdiag_mean"` |
| **Outlier costs** | `epsilon_mode="offdiag_median"` |
| **Conservative** | `epsilon_mode="offdiag_max"` |
| **Underfitting** | Decrease `epsilon_scale` (e.g., 0.5) |
| **Overfitting** | Increase `epsilon_scale` (e.g., 2.0) |
| **Controlled experiment** | `epsilon_mode="constant"` with fixed value |

### Convergence

- **Sinkhorn variants**: Typically converge in 50-100 iterations
- **Frank-Wolfe**: Typically converges in 30-50 iterations
- Monitor warnings about non-convergence (increase `max_iter` if needed)

---

## References

- **POT Library**: [pythonot.github.io](https://pythonot.github.io/)
- **Cuturi (2013)**: "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
- **Blondel et al. (2020)**: "Learning with Fenchel-Young Losses"
- **Fraud detection application**: See [`docs/fraud_business_and_cost_matrix.md`](fraud_business_and_cost_matrix.md)

---

## See Also

- [Main README](../README.md) — Usage examples and command-line interface
- [Fraud Cost Matrix Documentation](fraud_business_and_cost_matrix.md) — Business value modeling
- [SinkhornPOTLoss Example](../examples/sinkhorn_pot_example.py) — Standalone code example
