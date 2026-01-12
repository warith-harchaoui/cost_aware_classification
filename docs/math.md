# Mathematics: cost-aware losses in this repo

This repository implements three cost-aware losses for multi-class classification:

- **SinkhornFenchelYoungLoss** (implicit Fenchel–Young / Frank–Wolfe inner solve)
- **SinkhornEnvelopeLoss** (entropic OT + envelope gradient)
- **SinkhornFullAutodiffLoss** (entropic OT + full autodiff through Sinkhorn)

The common ingredients are:

- A **score vector** (logits) $f \in \mathbb{R}^K$
- A true label $y \in \{0,\dots,K-1\}$
- A **cost matrix** $C \in \mathbb{R}_+^{K\times K}$ where $C_{i,j}$ is the business cost of *acting as class* $j$ when the true class is $i$.
- We assume $C_{i,i}=0$ and $C_{i,j}\ge 0$.

All losses support **example-dependent costs**:
- $C$ can be global $(K,K)$ or batched $(B,K,K)$.

---

## Entropic OT: $\varepsilon$ vs POT's `reg`

Balanced entropic OT between distributions $p, q \in \Delta_K$ is often defined as:


```math
\mathrm{OT}_\varepsilon(p,q) \;=\; \min_{P \in \mathbb{R}_+^{K\times K}}
\;\langle P, C \rangle \;+\; \varepsilon\;\mathrm{KL}(P \,\|\, p\otimes q)
\quad\text{s.t.}\quad
P\mathbf{1} = p,\;\; P^\top \mathbf{1} = q.
```


The entropic regularization parameter is **the scalar multiplying the KL term**.

- In many papers, it is written $\varepsilon$.
- In **POT**, the same parameter is typically named `reg`.

**In this balanced KL-regularized formulation:**


```math
\boxed{\texttt{reg} = \varepsilon}
```


So an “apples-to-apples” comparison across implementations means matching this scalar.

---

## Sinkhorn algorithm (balanced)

Let the Gibbs kernel be:


```math
K \;=\; \exp(-C/\varepsilon).
```


Sinkhorn iterations solve for scaling vectors $u, v$ such that:


```math
P^\star = \mathrm{diag}(u)\,K\,\mathrm{diag}(v)
```


has marginals $p$ and $q$. The standard updates are:


```math
u \leftarrow \frac{p}{K v},\qquad
v \leftarrow \frac{q}{K^\top u},
```


with elementwise division.

---

## SinkhornEnvelopeLoss

### Forward
We build:

- $p = \mathrm{softmax}(f)$ (predicted distribution)
- $q = \mathrm{onehot}(y)$ (target distribution, with tiny smoothing for stability)

We compute a Sinkhorn plan $P^\star(p,q)$, then evaluate the **primal objective**:


```math
L(p,q) = \langle P^\star, C \rangle + \varepsilon\,\mathrm{KL}(P^\star \,\|\, p\otimes q).
```


### Backward (envelope / implicit-ish)
We treat $P^\star$ as **constant** in the backward pass (no differentiation through Sinkhorn iterations),
but we keep the explicit dependence on $p$ and $q$ through the KL term:

- gradients flow through $\log p$ (and $\log q$ if needed)
- gradients do **not** flow through the iterative solver

This matches the “don’t backprop through the solver” philosophy of Sinkhorn-Fenchel-Young.

---

## SinkhornFullAutodiffLoss

Same objective as above, but **we differentiate through the Sinkhorn iterations**.
This can be more “end-to-end”, but typically costs more memory and can be less stable for large iteration counts.

---

## SinkhornFenchelYoungLoss (Fenchel–Young)

Sinkhorn-Fenchel-Young implements a Fenchel–Young loss of the form:


```math
\ell(y,f) = \Omega^\star_{C,\varepsilon}(f) - f_y,
```


where $\Omega^\star$ is the Fenchel conjugate of a convex regularizer $\Omega$
that encodes the cost geometry.

Computing $\Omega^\star$ involves an inner optimization over the simplex (solved by Frank–Wolfe),
but the implementation uses an **implicit gradient** by computing the inner solution under `torch.no_grad()`.

This yields stable training dynamics while avoiding expensive differentiation through an inner loop.

---

## Practical notes

- $\varepsilon$ is a *scale parameter*: changing it changes the geometry / smoothness of the surrogate.
- For numerical stability, it is common to:
  - clamp probabilities in $\log(\cdot)$,
  - add very small **label smoothing** to the one-hot $q$.
