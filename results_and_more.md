https://docs.google.com/document/d/1X_PQ1DyON2m9epAmQxX8YIb39SxXDIj3GgJSjgyMSXU/edit?usp=sharing

# Sparsity
1. Issue Summary
The PyTorch Soft Decision Tree (SDT) student model exhibits severe performance degradation when utilizing mean imputation for missing values during knowledge distillation from the XGBoost teacher model. XGBoost natively employs sparsity-aware split finding, which treats missingness as a highly predictive feature by mathematically routing missing values down an optimal "default direction." Mean imputation neutralizes this critical signal by forcing the SDT to treat missing records as average observations.  2. Proposed Architecture ModificationTo successfully distill the sparsity-aware logic from the teacher model, the SDT architecture must be modified to incorporate "Learnable Default Directions." This requires replacing mean imputation with a dual-input strategy: zero-imputation combined with a concatenated binary missingness mask.3. Mathematical Mechanism
The standard SDT internal node calculates a routing probability using a sigmoid activation function over the input vector $x$, weight vector $w$, and bias $b$:  $$p = \sigma(w^T x + b)$$By applying zero-imputation to the missing values ($x_{zero}$) and concatenating a binary missingness mask ($m$), the routing calculation is expanded:$$p = \sigma(w^T x_{zero} + w_{mask}^T m + b)$$4. Expected BehaviorUnder the modified architecture, the network isolates the missingness signal:When a feature is missing, the value in $x_{zero}$ is $0.0$, effectively neutralizing the standard feature weight $w$.The corresponding value in the mask $m$ evaluates to $1.0$, activating the specialized, learnable weight $w_{mask}$.During backpropagation, the network independently optimizes $w_{mask}$ to act as the default direction. This weight can push the sigmoid activation heavily toward $0$ (Left child) or $1$ (Right child) based solely on the presence of the missingness indicator, thereby successfully replicating the XGBoost routing behavior.

# dGBDT Implementation Analysis

The dGBDT ensemble follows the **TNT (Tree-to-Neural-Network Transformation)** architecture, implementing an additive ensemble $F(x) = \sum \beta_m \sigma(h_m(x))$ with stage-wise residual boosting.

## Key Findings & Improvements

### 1. Beta Parameter Learning
A significant bug was identified in the initial training loop where the `beta` (tree weight) parameters were not learning. In the boosting loop, `beta[m]` was used to update the cumulative prediction (`pred_so_far`), but since `pred_so_far` was detached before being used in the next tree's residual calculation, the gradient never reached the `beta` weights.

**Fix:** The loss calculation was modified to include `beta[m]` inside the optimized MSE term at each stage:
`total_loss = total_loss + mse(beta[m] * torch.sigmoid(tree_output), residual)`

### 2. SDT Regularization Optimization
The `SDT` penalty calculation was transitioned to a vectorized implementation. This removes inefficient Python loops over nodes and resolves a variable shadowing issue in the penalty list generation (e.g., ensuring the `depth` parameter is not overwritten by loop variables).

### 3. Sparsity-Aware Routing
To successfully distill knowledge from an XGBoost teacher (which handles sparsity natively), the student SDTs utilize a dual-input strategy of zero-imputation combined with a concatenated binary missingness mask. This allows the network to learn "default directions" for missing values.



main.py - iseeu model 
main_tnt.py - with same dataset
we don't need imputation anymore
main_tnt.py - with sparse dataset (not so good)

de incercat:
    summary features in the dGBDT model

---

# TNT Pipeline — Bugs Found, Fixes Applied, and How the Algorithm Works

## 1. What Was Wrong and Why

Three interrelated algorithmic bugs were identified that together prevented the dGBDT from performing meaningful gradient boosting.

### Bug 1 — The Positivity Trap (most critical)

**What was wrong:** The ensemble prediction was computed as:
$$F(x) = \sum_m \beta_m \cdot \sigma(h_m(x))$$
where $\beta_m > 0$ (enforced via `softplus`) and $\sigma(\cdot) \in (0,1)$. Every tree contribution was strictly positive. This meant tree $m$ could only *increase* the cumulative prediction — it could never subtract from it. If tree 1 over-predicted mortality risk, no subsequent tree could correct that mistake. The model was not gradient boosting in any meaningful sense; it was gradient *accumulation*.

**Why it breaks boosting:** Classical GBDT works by fitting each new tree to the *negative gradient* (residual) of the loss. Residuals can be negative. A model that can only add positive values cannot reduce an over-prediction.

### Bug 2 — Residual Computed in the Wrong Space

**What was wrong:** The cumulative prediction `pred_so_far` was accumulating the full `beta * sigmoid(logit)` contributions (bounded in $(0, 1)$), but the residual was calculated as:
```python
residual = yb - pred_so_far
```
This subtracted a probability-valued prediction from a probability-valued label — which is technically consistent. However, once Bug 1 was fixed and `pred_so_far` became an accumulator of raw *logits* (unbounded), this formula broke completely: it subtracted an unbounded logit from a probability in $[0,1]$, making the residual numerically meaningless.

**Why it breaks:** Targets (`yb`) live in probability space $[0,1]$. Raw logit sums can be arbitrarily large or negative. You cannot subtract them directly.

### Bug 3 — Same Space Mismatch in Distillation

**What was wrong:** The distillation training loop in `distill_tnt_dgbdt.py` had the same residual mismatch. The teacher soft labels (`ytb`) are probabilities in $[0,1]$ output by the TNT-ConvNet's sigmoid. The mixed distillation residual was:
```python
residual = alpha * (ytb - pred_so_far) + (1 - alpha) * (yb - pred_so_far)
```
Again, subtracting the logit accumulator directly from probability-valued targets.

---

## 2. How They Were Fixed

### Fix for Bug 1 — Remove `beta` and `sigmoid` from the ensemble sum
The `PositiveParam` class and `self.beta` parameter were removed entirely. The ensemble now sums raw, unbounded logits:
$$F(x) = \sigma\!\left(\sum_m h_m(x)\right)$$
Each $h_m(x)$ is the raw scalar output of an SDT, which can be any real number. The sigmoid is applied only once, at the very end, to produce a probability. Later trees can now output negative values to reduce the cumulative logit.

### Fix for Bugs 2 & 3 — Apply `sigmoid` at the moment of residual computation
`pred_so_far` accumulates raw logits. The residual is computed by projecting the cumulative logit back into probability space before comparing it to the probability target:
```python
residual = yb - torch.sigmoid(pred_so_far.detach())
```
For distillation, both the teacher signal and the ground-truth label are handled consistently:
```python
residual = alpha * (ytb - torch.sigmoid(pred_so_far.detach())) \
         + (1 - alpha) * (yb - torch.sigmoid(pred_so_far.detach()))
```
This preserves the mathematical integrity of Algorithm 2 from the TNT paper: the cumulative output $s_{m-1}$ is the sum of raw logits, but the target comparison is always done in probability space.

---

## 3. How the Full Algorithm Works Now

### Stage 0 — Data Preprocessing (`preprocess.py`)

Raw MIMIC-III records are pivoted into a tensor of shape `(N_patients, 22_features, 48_hours)`. 22 clinical measurements (GCS components, vitals, labs, comorbidities) are recorded at each of the first 48 ICU hours. Missing values are left as `NaN` in the sparse dataset; `nan_to_num` with zero-fill plus a concatenated binary missingness mask is applied at the SDT input layer.

### Stage 1 — XGBoost Teacher: Temporal Feature Engineering

For each cross-validation fold, an XGBoost classifier is trained on the flattened hourly snapshots (`N × 48` rows of 22 features each, with each patient's label repeated 48 times). XGBoost uses sparsity-aware split finding — missing values are routed to a learned "default direction" at each node, so missingness itself becomes a predictive signal.

After training, the model is used in **leaf-prediction mode**: every patient-hour is passed through all 60 XGBoost trees and the integer leaf index from each tree is recorded. This produces a leaf-index tensor of shape `(N, 48, num_trees)`. These indices encode the decision path each hourly snapshot took — a compact, tree-structured representation of the clinical state at each hour.

### Stage 2 — TNT-ConvNet Teacher: Learning from Leaf Sequences

The leaf indices are passed into a Keras model with a custom `TreeEmbeddingLayer`. Each tree's leaf index is mapped to a learned dense vector of dimension 4 via an embedding table; embeddings from all trees at each hour are concatenated, producing a representation of shape `(N, 48, num_trees × 4)`.

This 2D temporal-feature map is treated as a single-channel image and fed into three parallel Conv2D branches with temporal kernel heights of 3, 6, and 12 (capturing 3h, 6h, and 12h patterns). The branches are concatenated, passed through BatchNorm and Dropout, then a dense head outputs a mortality probability. The model is trained with binary cross-entropy against ground-truth ICU mortality labels and class-weighted loss to handle class imbalance (~9:1 survive/die).

The best checkpoint per fold is saved. Its predictions on the validation set are the **teacher soft labels** for distillation.

### Stage 3 — dGBDT Student: Distillation via Stage-wise Residual Boosting

The student `dGBDT` in `dGBT.py` is a PyTorch ensemble of $M = 5$ Soft Decision Trees (SDTs). Each SDT is a differentiable binary tree of depth 5 with $2^5 = 32$ leaves.

**SDT internals:** Each internal node computes a probabilistic routing:
$$p_{\text{left}} = \sigma\!\left(W_{\text{inner}}^T \, [1;\, x_{\text{zero}};\, m_{\text{mask}}]\right)$$
The input is augmented with a constant bias term and a binary missingness mask (double-width input). The probability of reaching each leaf is the product of routing decisions along the path (left = $p$, right = $1-p$). The tree output is the weighted sum of leaf scalar weights:
$$h_m(x) = \sum_\ell \mu_\ell(x) \cdot \phi_\ell$$
where $\mu_\ell$ is the path probability to leaf $\ell$ and $\phi_\ell \in \mathbb{R}$ is the leaf's learned scalar. This output is unbounded.

**Training loop (Algorithm 2, TNT paper):** For each mini-batch:

1. Obtain raw logit outputs $h_1(x), \ldots, h_M(x)$ from all SDTs simultaneously.
2. Initialise cumulative logit $s_0 = 0$.
3. For $m = 1, \ldots, M$:
   - Compute the mixed residual:
     $$r_m = \alpha \cdot T^2 \cdot \bigl(t - \sigma(s_{m-1})\bigr) + (1-\alpha) \cdot \bigl(y - \sigma(s_{m-1})\bigr)$$
     where $t$ is the teacher probability, $y \in \{0,1\}$ is the hard label, $\alpha = 0.5$, $T = 1.0$ is the distillation temperature.
   - Fit tree $m$ to this residual: $\mathcal{L}_m = \text{MSE}(h_m(x),\; r_m) + \text{penalty}_m$
   - Update the cumulative logit: $s_m = s_{m-1} + h_m(x)$
4. Total loss: $\mathcal{L} = \sum_m \mathcal{L}_m$, backpropagated through all SDT parameters jointly.

**Final prediction:** $\hat{y} = \sigma\!\left(\sum_m h_m(x)\right)$

### Input Flow Summary

```
MIMIC-III raw CSVs
      │
      ▼
  preprocess.py
  (N, 22, 48) with NaNs  ──────────────────────────────────┐
      │                                                     │ (for dGBDT)
      ▼                                                     │ flatten + zero-fill + mask
  XGBoost teacher                                           │ → (N, 1056) float32
  (trained on hourly snapshots)                             │
      │                                                     │
      ▼                                                     │
  Leaf index tensor (N, 48, num_trees)                      │
      │                                                     │
      ▼                                                     │
  TreeEmbeddingLayer                                        │
  (N, 48, num_trees × 4)                                    │
      │                                                     │
      ▼                                                     │
  TNT-ConvNet (3/6/12h parallel Conv2D)                     │
  → mortality probability ∈ (0,1)   [teacher soft label]   │
      │                                                     │
      └──────────────┬──────────────────────────────────────┘
                     ▼
             dGBDT distillation
             5 SDTs, depth 5
             stage-wise residual boosting
             mixed loss: α·teacher + (1-α)·hard label
                     │
                     ▼
             final prediction σ(Σ hₘ(x)) ∈ (0,1)
```





### The distilled student is too powerful, does not make sense
student on aggregated features -> BEST

