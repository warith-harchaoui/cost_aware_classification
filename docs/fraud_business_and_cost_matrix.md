# Fraud business framing and cost matrix construction (IEEE-CIS / Kaggle)

This note explains how to construct an **instance-dependent regret (cost) matrix** for the
binary fraud decision problem (approve vs decline), starting from a **business value model**.
It follows the “geometry of regret” principle:

> Start with a value matrix $V$ (truth $\times$ decision) in meaningful units (€, \$, minutes, …),
> then convert values to **regrets/costs** by subtracting the best achievable value for each truth state.

The goal is to obtain a cost matrix $C$ that is:
- **economically interpretable** (units = money),
- **zero on the diagonal** (no regret when choosing the best action for that truth),
- **instance-dependent** through the transaction amount $M$ (and possibly other attributes).

---

## 1) Setup: labels, actions, and notation

We consider **binary fraud classification**:

- Truth (label) $y \in \{0,1\}$
  - $y=0$: **legit**
  - $y=1$: **fraud**

- Actions / decisions $a \in \{0,1\}$
  - $a=0$: **approve**
  - $a=1$: **decline**

- Transaction amount (in currency units): $M > 0$

We will build:
- a **value matrix** $V(M)\in\mathbb{R}^{2\times 2}$, where $V_{y,a}(M)$ is the business value when truth is $y$ and we take decision $a$,
- then the **regret/cost matrix** $C(M)\in\mathbb{R}_+^{2\times 2}$.

---

## 2) The geometry-of-regret rule (values → regrets)

Given any value matrix $V\in\mathbb{R}^{K\times K}$ (here $K=2$), define for each truth state $y$:

```math
a^\*(y)\in \arg\max_{a} V_{y,a}
```

and define the regret/cost matrix:

```math
C_{y,a} = V_{y,a^\*(y)} - V_{y,a} \;\;\ge 0.
```

Key consequences:
- $C_{y,a^\*(y)} = 0$ by construction (no regret when choosing the best action under truth $y$),
- regrets share the same units as the values (e.g., euros),
- in many classification settings, $a^\*(y)=y$ so the cost matrix is **zero-diagonal**.

---

## 3) A practical business value model for approve/decline

We model two business effects:

### 3.1 Legit transaction (truth = legit)

If the transaction is legit, approving typically yields **positive value** (profit / contribution margin).
Declining incurs **friction / lost margin** (lost customer, lost sale, dissatisfaction).

A simple model:

- Approve (good outcome):
  ```math
  V_{\text{legit},\,\text{approve}}(M) = M.
  ```
  This “$M$” can be interpreted as a *normalized* positive value proportional to amount.
  (If you prefer “profit margin”, replace $M$ by $m\cdot M$ where $m\in(0,1)$.)

- Decline (bad outcome):
  ```math
  V_{\text{legit},\,\text{decline}}(M) = -\rho_{FD} \, M
  ```
  where $\rho_{FD}\ge 0$ captures the *relative friction / foregone value* from declining a legit transaction.
  This captures lost margin, customer churn, support cost, etc.

### 3.2 Fraudulent transaction (truth = fraud)

If the transaction is fraudulent, approving causes **fraud loss**;
declining avoids that loss.

- Decline (good outcome):
  ```math
  V_{\text{fraud},\,\text{decline}}(M) = 0.
  ```

- Approve (bad outcome):
  ```math
  V_{\text{fraud},\,\text{approve}}(M) = -L_{\text{fraud}}(M),
  ```
  where $L_{\text{fraud}}(M)\ge 0$ is the expected fraud loss **if approved**.

A common, reasonable instantiation is:
```math
L_{\text{fraud}}(M) = \lambda_{cb} M + F_{cb},
```
where:
- $\lambda_{cb}\ge 1$ is a chargeback multiplier (principal + fees + operational overhead),
- $F_{cb}\ge 0$ is a fixed dispute/chargeback fee (optional but often realistic).

---

## 4) Value matrix $V(M)$

Putting the above together:

```math
V(M) \;=\;
\begin{array}{c|cc}
\text{Reality}\backslash\text{Action} & \text{approve} & \text{decline}\\ \hline
\text{legit} & M & -\rho_{FD} M \\
\text{fraud} & -L_{\text{fraud}}(M) & 0
\end{array}
```

Interpretation:
- For legit: approving is best (typically), declining is harmful by a factor $\rho_{FD}M$.
- For fraud: declining is best (value 0), approving is harmful by $L_{\text{fraud}}(M)$.

---

## 5) Induced regret/cost matrix $C(M)$

Apply the rule:

- If truth is legit: $a^\*(\text{legit})=\text{approve}$ (since $M > -\rho_{FD}M$)
  ```math
  C_{\text{legit},\,\text{approve}} = M - M = 0
  ```
  ```math
  C_{\text{legit},\,\text{decline}} = M - (-\rho_{FD}M) = (1+\rho_{FD})M
  ```

- If truth is fraud: $a^\*(\text{fraud})=\text{decline}$ (since $0 > -L_{\text{fraud}}(M)$)
  ```math
  C_{\text{fraud},\,\text{approve}} = 0 - (-L_{\text{fraud}}(M)) = L_{\text{fraud}}(M)
  ```
  ```math
  C_{\text{fraud},\,\text{decline}} = 0 - 0 = 0
  ```

Therefore:

```math
C(M) \;=\;
\begin{array}{c|cc}
\text{Reality}\backslash\text{Action} & \text{approve} & \text{decline}\\ \hline
\text{legit} & 0 & (1+\rho_{FD})M \\
\text{fraud} & L_{\text{fraud}}(M) & 0
\end{array}
```

This is the matrix you should implement per transaction.

---

## 6) Mapping to ML tensor indices

If you represent:
- class/label index 0 = legit, 1 = fraud
- prediction/action index 0 = approve, 1 = decline

then the per-example cost tensor $C_i\in\mathbb{R}^{2\times 2}$ should be:

```math
C_i =
\begin{pmatrix}
0 & (1+\rho_{FD})M_i\\
L_{\text{fraud}}(M_i) & 0
\end{pmatrix}.
```

This matches the “zero diagonal, meaningful off-diagonal regrets” principle.

---

## 7) Decision rule induced by the cost matrix (deployment)

Given a model’s predicted probability $p_i = \mathbb{P}(y=1\mid x_i)$ for fraud:

- Expected cost if **approve**:
  ```math
  \mathbb{E}[C \mid a=\text{approve}] = p_i \cdot L_{\text{fraud}}(M_i).
  ```

- Expected cost if **decline**:
  ```math
  \mathbb{E}[C \mid a=\text{decline}] = (1-p_i) \cdot (1+\rho_{FD})M_i.
  ```

So decline iff:
```math
(1-p_i)(1+\rho_{FD})M_i < p_i L_{\text{fraud}}(M_i).
```

If $L_{\text{fraud}}(M)=\lambda_{cb}M$ and $F_{cb}=0$, the $M$ cancels and the threshold becomes constant:
```math
p_i > \frac{1+\rho_{FD}}{(1+\rho_{FD})+\lambda_{cb}}.
```

If you include a fixed fee $F_{cb}>0$, the threshold becomes **amount-dependent** (more realistic).

---

## 8) Reasonable benchmark constants for IEEE-CIS (Kaggle) experiments

The Kaggle dataset does not come with your true merchant economics, so we choose **sane, defensible defaults**
to create a reproducible benchmark. These are not “truth”; they are a reasonable *simulation* of business costs.

### 8.1 Choose the units
The dataset’s `TransactionAmt` has currency-like magnitude. Treat it as **dollars/euros** for benchmark purposes.

### 8.2 Recommended constants (baseline)
These values are intended to be:
- stable numerically,
- plausible economically,
- easy to sweep.

**Friction / foregone value from false decline**
- $\rho_{FD} = 0.10$  
  Interpretation: declining a legit transaction costs about 10% of the amount *in addition* to losing the sale itself
  under the value model used above (hence the $(1+\rho_{FD})M$ regret).

**Fraud loss model**
- $\lambda_{cb} = 1.5$  
  Interpretation: expected fraud loss is about 1.5× the amount (amount + fees + ops overhead).
- $F_{cb} = 15$  
  Interpretation: fixed dispute/chargeback fee (helps make the decision threshold amount-dependent).

So:
```math
L_{\text{fraud}}(M)=1.5M + 15.
```

These are a good starting point; you can sweep $\rho_{FD}\in\{0.05,0.10,0.20\}$ and
$\lambda_{cb}\in\{1.0,1.5,2.0,3.0\}$, $F_{cb}\in\{0,10,25\}$.

### 8.3 If you want the old “manual review cost” framing
Your previous script used a constant FP “review cost” (e.g., 5€). That corresponds to a *different action set*:
approve vs **review** vs decline, or decline meaning “manual review”.  
In the strict approve/decline model above, a constant review cost is not directly aligned unless you interpret
decline as “send to review”.

If you want “decline = review”, replace the legit-decline regret by a constant:
```math
C_{\text{legit},\,\text{decline}} = c_{FP}.
```
But note this deviates from the value-matrix derivation and removes amount dependence on the false-decline side.

For apples-to-apples comparisons with the regret geometry, prefer the $(1+\rho_{FD})M$ form.

---

## 9) What to use in experiments (CE, CACIS, Sinkhorn envelope/autodiff)

### 9.1 Training losses
- **CrossEntropyLoss**: cost-agnostic baseline, still useful.
- **SinkhornFenchelYoungLoss**: consumes the per-example cost matrix $C_i$.
- **SinkhornEnvelopeLoss**: consumes $C_i$; uses POT Sinkhorn with envelope gradients.
- **SinkhornFullAutodiffLoss**: consumes $C_i$; uses POT Sinkhorn with unrolled gradients.

### 9.2 Evaluation metrics (recommended)
For validation, report at least:
1) mean **realized regret** $C_{y_i,\hat y_i}$ using the cost-optimal decision rule,
2) mean **expected optimal regret** $\min\big(p_i L_{\text{fraud}}(M_i), (1-p_i)(1+\rho_{FD})M_i\big)$,
3) PR-AUC (sanity metric).

---

## 10) Summary (implementation checklist)

For each transaction $i$ with amount $M_i$:

1. Choose constants $\rho_{FD}, \lambda_{cb}, F_{cb}$.
2. Compute:
   - $c_{FD,i} = (1+\rho_{FD})M_i$
   - $c_{CB,i} = L_{\text{fraud}}(M_i)=\lambda_{cb}M_i + F_{cb}$
3. Build:
```math
C_i =
\begin{pmatrix}
0 & c_{FD,i}\\
c_{CB,i} & 0
\end{pmatrix}.
```

That is the **instance-dependent cost matrix** consistent with the geometry-of-regret construction.
