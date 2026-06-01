import torch
import torch.nn as nn

class SDT(nn.Module):
    """Fast, corrected implementation of soft decision tree in PyTorch."""

    def __init__(
            self,
            input_dim,
            output_dim,
            depth=5,
            lamda=1e-3,
            use_cuda=False):
        super(SDT, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.depth = depth
        self.lamda = lamda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self._validate_parameters()

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        # FIXED: Changed variable name from `depth` to `d` to avoid shadowing
        self.penalty_list = [
            self.lamda * (2 ** (-d)) for d in range(0, self.depth)
        ]

        self.inner_nodes = nn.Sequential(
            nn.Linear((self.input_dim * 2) + 1, self.internal_node_num_, bias=False),
            nn.Sigmoid(),
        )

        # FIXED: Leaf nodes are independent learnable parameters, not a Linear layer.
        # Initialized with small random values.
        self.leaf_parameters = nn.Parameter(
            torch.randn(self.leaf_node_num_, self.output_dim) * 0.1
        )

    def forward(self, X, is_training_data=False):
        _mu, _penalty = self._forward(X)
        
        # FIXED: Apply Sigmoid (for binary classification) to the leaves FIRST.
        # Now every leaf outputs a valid probability between 0 and 1.
        leaf_probs = torch.sigmoid(self.leaf_parameters)
        
        # y_pred = sum(mu_l * Q^l)
        # _mu shape: [Batch, Leaves]
        # leaf_probs shape: [Leaves, Output_Dim]
        y_pred = torch.matmul(_mu, leaf_probs)

        if is_training_data:
            return y_pred, _penalty
        else:
            return y_pred

    def _forward(self, X):
        batch_size = X.size()[0]
        X = self._data_augment(X)

        path_prob = self.inner_nodes(X)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        _mu = torch.ones(batch_size, 1, 1, device=self.device)
        # TWEAK: slightly cleaner device assignment
        _penalty = torch.tensor(0.0, device=self.device)

        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            _penalty = _penalty + self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)

            _mu = _mu * _path_prob

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(batch_size, self.leaf_node_num_)

        return mu, _penalty

    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        eps = 1e-6
        batch_size = _mu.size(0)
        
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob_left = _path_prob.view(batch_size, -1)[:, 0::2]

        numerator = (_path_prob_left * _mu).sum(dim=0)               
        denominator = _mu.sum(dim=0).clamp(min=eps)                  
        alpha = (numerator / denominator).clamp(min=eps, max=1.0 - eps)

        coeff = self.penalty_list[layer_idx]
        return -0.5 * coeff * (torch.log(alpha) + torch.log(1 - alpha)).sum()

    def _data_augment(self, X):
        batch_size = X.size(0)
        X = X.view(batch_size, -1)

        missing_mask = torch.isnan(X).float()
        X_zero_filled = torch.nan_to_num(X, nan=0.0)
        X_combined = torch.cat((X_zero_filled, missing_mask), dim=1)

        bias = torch.ones(batch_size, 1, device=X.device)
        return torch.cat((bias, X_combined), 1)

    def _validate_parameters(self):
        if not self.depth > 0:
            raise ValueError(f"The tree depth should be strictly positive, but got {self.depth} instead.")
        if not self.lamda >= 0:
            raise ValueError(f"The coefficient of the regularization term should not be negative, but got {self.lamda} instead.")