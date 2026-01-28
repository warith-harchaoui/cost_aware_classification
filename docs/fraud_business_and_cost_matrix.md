# Fraud Business Framing and Cost Matrix Construction (IEEE-CIS / Kaggle)

**Author:** Warith Harchaoui <wharchaoui@nexton-group.com>

This document explains how to construct an **instance-dependent regret (cost) matrix** for the binary fraud decision problem (approve vs decline), starting from a **business value model**.

## Philosophy: Geometry of Regret

We follow the "geometry of regret" principle:

> Start with a value matrix $V$ (truth × decision) in meaningful units (€, \$, minutes, …),  
> then convert values to **regrets/costs** by subtracting the best achievable value for each truth state.

The goal is to obtain a cost matrix $C$ that is:
- **Economically interpretable** (units = money)
- **Zero on the diagonal** (no regret when choosing the best action for that truth)
- **Instance-dependent** through transaction amount $M$ (and possibly other attributes)

---

## 1. Setup: Labels, Actions, and Notation

We consider **binary fraud classification**:

**Truth (label)** $y \in \{0,1\}$:
- $y=0$: **legit** transaction
- $y=1$: **fraud** transaction

**Actions / decisions** $a \in \{0,1\}$:
- $a=0$: **approve** transaction
- $a=1$: **decline** transaction

**Transaction amount** (in currency units): $M > 0$

We will build:
- A **value matrix** $V(M)\in\mathbb{R}^{2\times 2}$, where $V_{y,a}(M)$ is the business value when truth is $y$ and we take decision $a$
- Then the **regret/cost matrix** $C(M)\in\mathbb{R}_+^{2\times 2}$

---

##  2. The Geometry-of-Regret Rule (Values → Regrets)

Given any value matrix $V\in\mathbb{R}^{K\times K}$ (here $K=2$), define for each truth state $y$:

```math
a^*(y)\in \arg\max_{a} V_{y,a}
```

and define the regret/cost matrix:

```math
C_{y,a} = V_{y,a^*(y)} - V_{y,a} \;\;\ge 0
```

**Key consequences:**
- $C_{y,a^*(y)} = 0$ by construction (no regret when choosing the best action under truth $y$)
- Regrets share the same units as the values (e.g., euros)
- In many classification settings, $a^*(y)=y$ so the cost matrix is **zero-diagonal**

---

## 3. A Practical Business Value Model for Approve/Decline

We model two business effects:

### 3.1 Legit Transaction (Truth = Legit)

If the transaction is legit, approving typically yields **positive value** (profit / contribution margin).  
Declining incurs **friction / lost margin** (lost customer, lost sale, dissatisfaction).

**Simple model:**

**Approve** (good outcome):
```math
V_{\text{legit},\,\text{approve}}(M) = M
```

This "$M$" can be interpreted as a *normalized* positive value proportional to amount.  
(If you prefer "profit margin", replace $M$ by $m\cdot M$ where $m\in(0,1)$.)

**Decline** (bad outcome):
```math
V_{\text{legit},\,\text{decline}}(M) = M -\rho_{FD} \, M
```

where $\rho_{FD}\in [0, 1]$ captures the *relative friction / foregone value* from declining a legit transaction.
  
This captures lost margin, customer churn, support cost, etc.

**Typical value:** $\rho_{FD} = 0.10$ (10% additional loss beyond the sale itself)

### 3.2 Fraudulent Transaction (Truth = Fraud)

If the transaction is fraudulent, approving causes **fraud loss**; declining avoids that loss.

**Decline** (good outcome):
```math
V_{\text{fraud},\,\text{decline}}(M) = 0
```

**Approve** (bad outcome):
```math
V_{\text{fraud},\,\text{approve}}(M) = -L_{\text{fraud}}(M)
```

where $L_{\text{fraud}}(M)\ge 0$ is the expected fraud loss **if approved**.

**Common, reasonable instantiation:**
```math
L_{\text{fraud}}(M) = \lambda_{cb} M + F_{cb}
```

where:
- $\lambda_{cb}\ge 1$: chargeback multiplier (principal + fees + operational overhead)
- $F_{cb}\ge 0$: fixed dispute/chargeback fee

**Typical values:**
- $\lambda_{cb} = 1.5$ (1.5× the amount)
- $F_{cb} = 15$ (fixed $15 fee)

---

## 4. Value Matrix $V(M)$

Putting the above together:

```math
V(M) \;=\;
\begin{array}{c|cc}
\text{Reality}\backslash\text{Action} & \text{approve} & \text{decline}\\ \hline
\text{legit} & M & M -\rho_{FD} M \\
\text{fraud} & -L_{\text{fraud}}(M) & 0
\end{array}
```

**Interpretation:**
- For legit: approving is best (typically), declining is harmful by a factor $\rho_{FD}M$ (corresponding to friction because of being annoyed by verification)
- For fraud: declining is best (value 0), approving is harmful by $L_{\text{fraud}}(M)$

---

## 5. Induced Regret/Cost Matrix $C(M)$

Apply the geometry-of-regret rule:

**If truth is legit:** $a^*(\text{legit})=\text{approve}$ (since $M > M -\rho_{FD}M$ for $\rho_{FD}>0$)

```math
C_{\text{legit},\,\text{approve}} = M - M = 0
```

```math
C_{\text{legit},\,\text{decline}} = M - (M -\rho_{FD}M) = \rho_{FD}M
```

**If truth is fraud:** $a^*(\text{fraud})=\text{decline}$ (since $0 > -L_{\text{fraud}}(M)$)

```math
C_{\text{fraud},\,\text{approve}} = 0 - (-L_{\text{fraud}}(M)) = L_{\text{fraud}}(M)
```

```math
C_{\text{fraud},\,\text{decline}} = 0 - 0 = 0
```

**Therefore:**

```math
C(M) \;=\;
\begin{pmatrix}
0 & \rho_{FD}M \\
L_{\text{fraud}}(M) & 0
\end{pmatrix}
```

This is the matrix you should implement per transaction.

---

## 6. Mapping to ML Tensor Indices

If you represent:
- Class/label index 0 = legit, 1 = fraud
- Prediction/action index 0 = approve, 1 = decline

then the per-example cost tensor $C_i\in\mathbb{R}^{2\times 2}$ should be:

```math
C_i =
\begin{pmatrix}
0 & \rho_{FD}M_i\\
L_{\text{fraud}}(M_i) & 0
\end{pmatrix}
```

This matches the "zero diagonal, meaningful off-diagonal regrets" principle.

**Implementation example:**
```python
import numpy as np

def build_cost_matrix(amount: np.ndarray, 
                     rho_fd: float = 0.10,
                     lambda_cb: float = 1.5,
                     F_cb: float = 15.0) -> np.ndarray:
    """
    Build per-example cost matrices for fraud detection.
    
    Parameters
    ----------
    amount : np.ndarray
        Transaction amounts, shape (N,)
    rho_fd : float
        False decline friction parameter (default: 0.10)
    lambda_cb : float
        Chargeback multiplier (default: 1.5)
    F_cb : float
        Fixed chargeback fee (default: 15.0)
    
    Returns
    -------
    np.ndarray
        Cost matrices, shape (N, 2, 2)
    """
    M = amount.astype(np.float32)
    c_fd = rho_fd * M  # Cost of declining legit
    c_cb = lambda_cb * M + F_cb  # Cost of approving fraud
    
    C = np.zeros((M.shape[0], 2, 2), dtype=np.float32)
    C[:, 0, 1] = c_fd  # legit, decline
    C[:, 1, 0] = c_cb  # fraud, approve
    return C
```

---

## 7. Decision Rule Induced by the Cost Matrix (Deployment)

Given a model's predicted probability $p_i = \mathbb{P}(y=1\mid x_i)$ for fraud:

**Expected cost if approve:**
```math
\mathbb{E}[C \mid a=\text{approve}] = p_i \cdot L_{\text{fraud}}(M_i)
```

**Expected cost if decline:**
```math
\mathbb{E}[C \mid a=\text{decline}] = (1-p_i) \cdot \rho_{FD}M_i
```

**Decision rule: Decline if and only if:**
```math
(1-p_i)\rho_{FD}M_i < p_i L_{\text{fraud}}(M_i)
```

Rearranging:
```math
p_i > \frac{\rho_{FD}M_i}{\rho_{FD}M_i + L_{\text{fraud}}(M_i)}
```

**Special case:** If $L_{\text{fraud}}(M)=\lambda_{cb}M$ and $F_{cb}=0$, the $M$ cancels and the threshold becomes **constant:**

```math
p_i > \frac{\rho_{FD}}{\rho_{FD}+\lambda_{cb}}
```

**With $F_{cb}>0$:** The threshold becomes **amount-dependent** (more realistic for fraud detection).

---

## 8. Reasonable Benchmark Constants for IEEE-CIS (Kaggle) Experiments

The Kaggle dataset does not come with your true merchant economics, so we choose **sane, defensible defaults** to create a reproducible benchmark. These are not "truth"; they are a reasonable *simulation* of business costs.

### 8.1 Choose the Units

The dataset's `TransactionAmt` has currency-like magnitude. Treat it as **dollars/euros** for benchmark purposes.

### 8.2 Recommended Constants (Baseline)

These values are:
- Numerically stable
- Economically plausible
- Easy to sweep for sensitivity analysis

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| $\rho_{FD}$ | **0.10** | Declining a legit transaction costs ~10% additional loss beyond the sale |
| $\lambda_{cb}$ | **1.5** | Fraud loss is ~1.5× the amount (principal + fees + ops) |
| $F_{cb}$ | **15** | Fixed $15 dispute/chargeback fee |

**Fraud loss formula:**
```math
L_{\text{fraud}}(M)=1.5M + 15
```

### 8.3 Sensitivity Analysis

You can sweep these parameters to understand robustness:
- $\rho_{FD}\in\{0.05, 0.10, 0.20\}$
- $\lambda_{cb}\in\{1.0, 1.5, 2.0, 3.0\}$
- $F_{cb}\in\{0, 10, 25\}$

### 8.4 Alternative: Manual Review Cost Framing

Your previous script may have used a constant FP "review cost" (e.g., $5). That corresponds to a *different action set*: approve vs **review** vs decline.

If you want "decline = review", replace the legit-decline regret by a constant:
```math
C_{\text{legit},\,\text{decline}} = c_{FP}
```

**Note:** This deviates from the value-matrix derivation and removes amount dependence on the false-decline side. For apples-to-apples comparisons with the regret geometry, prefer the $\rho_{FD}M$ form.

---

## 9. What to Use in Experiments

### 9.1 Training Losses

This repository implements the following losses for cost-aware training:

1. **CrossEntropyLoss** (baseline)
   - Cost-agnostic
   - Useful for comparison

2. **Cross-Entropy Weighted**  
   - Uses sample weights $w_i = C_i[y_i, 1-y_i]$
   - Simple cost-aware baseline

3. **SinkhornFenchelYoungLoss**
   - Consumes per-example cost matrix $C_i$
   - Implicit differentiation via Frank-Wolfe
   - Provably stable gradients

4. **SinkhornEnvelopeLoss**
   - Custom Sinkhorn solver
   - Envelope gradients (no backprop through solver)
   - Memory efficient

5. **SinkhornFullAutodiffLoss**
   - Custom Sinkhorn solver
   - Full autodiff through iterations
   - Higher memory usage

6. **SinkhornPOTLoss** ⭐
   - Uses POT library's optimized Sinkhorn
   - Envelope gradients
   - **Recommended for production**
   - Best numerical stability

**See:** [`README.md`](../README.md) for detailed usage examples.

### 9.2 Evaluation Metrics (Recommended)

For validation, report at least:

1. **Mean realized regret:** $C_{y_i,\hat y_i}$ using the cost-optimal decision rule (the lower, the better)
2. **Mean expected optimal regret:** $\min\big(p_i L_{\text{fraud}}(M_i), (1-p_i)(1+\rho_{FD})M_i\big)$ (the lower, the better)
3. **PR-AUC** (sanity metric for class imbalance, the higher, the better)

**Example metrics output:**
```
PR-AUC: 0.847
Expected Optimal Regret: 12.34 €
Realized Regret: 14.56 €
```

Lower regret = better cost-aware performance.

---

## 10. Summary (Implementation Checklist)

For each transaction $i$ with amount $M_i$:

**1. Choose constants:**
-  $\rho_{FD} = 0.10$
- $\lambda_{cb} = 1.5$
- $F_{cb} = 15$

**2. Compute regrets:**
```python
c_fd_i = rho_fd * M_i        # False decline cost
c_cb_i = lambda_cb * M_i + F_cb    # Chargeback cost
```

**3. Build cost matrix:**
```math
C_i =
\begin{pmatrix}
0 & c_{FD,i}\\
c_{CB,i} & 0
\end{pmatrix}
```

**4. Use in training:**
```python
from cost_aware_losses import SinkhornPOTLoss

loss_fn = SinkhornPOTLoss(epsilon_mode="offdiag_median")
loss = loss_fn(scores, targets, C=cost_matrices)
loss.backward()
```

**5. Deploy with optimal decision rule:**
```python
def optimal_decision(fraud_prob, amount, rho_fd=0.10, lambda_cb=1.5, F_cb=15):
    """Return True if should decline."""
    cost_approve = fraud_prob * (lambda_cb * amount + F_cb)
    cost_decline = (1 - fraud_prob) * rho_fd * amount
    return cost_decline < cost_approve
```

This is the **instance-dependent cost matrix** consistent with the geometry-of-regret construction.

---

## References

- **IEEE-CIS Fraud Detection:** [Kaggle Competition](https://www.kaggle.com/c/ieee-fraud-detection)
- **Mathematical foundations:** [`docs/math.md`](math.md)
- **Usage examples:** [`README.md`](../README.md)
- **Code example:** [`examples/fraud_detection.py`](../examples/fraud_detection.py)

---

## Practical Tips

### Cost Matrix Validation

Always validate your cost matrices:
```python
# Check diagonal is zero
assert np.allclose(C[:, 0, 0], 0)
assert np.allclose(C[:, 1, 1], 0)

# Check non-negative
assert np.all(C >= 0)

# Check amount dependency
assert np.all(C[:, 0, 1] > 0)  # Decline cost > 0
assert np.all(C[:, 1, 0] > 0)  # Fraud cost > 0
```

### Debugging

If training seems unstable:
1. Verify cost matrix construction
2. Check for NaN/Inf in costs
3. Normalize amounts if scale is too large
4. Use `epsilon_mode="offdiag_median"` for robustness
5. Increase `epsilon_scale` for smoother regularization

### Performance Monitoring

Track both classification and cost metrics:
```python
results = {
    'pr_auc': 0.847,                  # (the higher, the better)
    'expected_opt_regret_euro': 12.34, # (the lower, the better)
    'realized_regret_euro': 14.56,     # (the lower, the better)
    'regret_vs_optimal_ratio': 1.18,  # (the lower, the better)
}
```

The ratio helps understand how close your model is to optimal cost-aware decisions.
