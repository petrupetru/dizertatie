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