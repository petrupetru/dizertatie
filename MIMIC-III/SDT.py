import torch
import torch.nn as nn


class SDT(nn.Module):
    """Fast implementation of soft decision tree in PyTorch.

    Parameters
    ----------
    input_dim : int
      The number of input dimensions.
    output_dim : int
      The number of output dimensions. For example, for a multi-class
      classification problem with `K` classes, it is set to `K`.
    depth : int, default=5
      The depth of the soft decision tree. Since the soft decision tree is
      a full binary tree, setting `depth` to a large value will drastically
      increases the training and evaluating cost.
    lamda : float, default=1e-3
      The coefficient of the regularization term in the training loss. Please
      refer to the paper on the formulation of the regularization term.
    use_cuda : bool, default=False
      When set to `True`, use GPU to fit the model. Training a soft decision
      tree using CPU could be faster considering the inherent data forwarding
      process.

    Attributes
    ----------
    internal_node_num_ : int
      The number of internal nodes in the tree. Given the tree depth `d`, it
      equals to :math:`2^d - 1`.
    leaf_node_num_ : int
      The number of leaf nodes in the tree. Given the tree depth `d`, it equals
      to :math:`2^d`.
    penalty_list : list
      A list storing the layer-wise coefficients of the regularization term.
    inner_nodes : torch.nn.Sequential
      A container that simulates all internal nodes in the soft decision tree.
      The sigmoid activation function is concatenated to simulate the
      probabilistic routing mechanism.
    leaf_nodes : torch.nn.Linear
      A `nn.Linear` module that simulates all leaf nodes in the tree.
    """

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

        # Different penalty coefficients for nodes in different layers
        self.penalty_list = [
            self.lamda * (2 ** (-d)) for d in range(0, self.depth)
        ]

        # Initialize internal nodes and leaf nodes, the input dimension on
        # internal nodes is added by 1, serving as the bias.
        self.inner_nodes = nn.Sequential(
            nn.Linear((self.input_dim * 2) + 1, self.internal_node_num_, bias=False),
            nn.Sigmoid(),
        )

        self.leaf_nodes = nn.Linear(self.leaf_node_num_,
                                    self.output_dim,
                                    bias=False)

    def forward(self, X, is_training_data=False):
        _mu, _penalty = self._forward(X)
        y_pred = self.leaf_nodes(_mu)

        # When `X` is the training data, the model also returns the penalty
        # to compute the training loss.
        if is_training_data:
            return y_pred, _penalty
        else:
            return y_pred

    def _forward(self, X):
        """Implementation on the data forwarding process."""

        batch_size = X.size()[0]
        X = self._data_augment(X)

        path_prob = self.inner_nodes(X)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        _mu = torch.ones(batch_size, 1, 1, device=self.device)
        _penalty = torch.tensor(0.0, device=self.device)

        # Iterate through internal nodes in each layer to compute the final path
        # probabilities and the regularization term.
        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
          _path_prob = path_prob[:, begin_idx:end_idx, :]

          # Extract internal nodes in the current layer to compute the
          # regularization term
          _penalty = _penalty + self._cal_penalty(layer_idx, _mu, _path_prob)
          _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)

          _mu = _mu * _path_prob  # update path probabilities

          begin_idx = end_idx
          end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(batch_size, self.leaf_node_num_)

        return mu, _penalty

    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        """
        Compute the regularization term for internal nodes in different layers.
        """
        penalty = torch.tensor(0.0).to(self.device)
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
        """Add a constant input `1` onto the front of each sample."""
        batch_size = X.size()[0]
        X = X.view(batch_size, -1)

        # 1. Create a binary mask of the missing values (1 if NaN, 0 otherwise)
        missing_mask = torch.isnan(X).float()

        # 2. Replace NaNs with 0.0 so the network can safely multiply them
        X_zero_filled = torch.nan_to_num(X, nan=0.0)

        # 3. Concatenate the features and the mask
        # Shape becomes (batch_size, input_dim * 2)
        X_combined = torch.cat((X_zero_filled, missing_mask), dim=1)

        bias = torch.ones(batch_size, 1).to(self.device)
        X = torch.cat((bias, X_combined), 1)

        return X

    def _validate_parameters(self):

        if not self.depth > 0:
            msg = ("The tree depth should be strictly positive, but got {}"
                   "instead.")
            raise ValueError(msg.format(self.depth))

        if not self.lamda >= 0:
            msg = (
                "The coefficient of the regularization term should not be"
                " negative, but got {} instead."
            )
            raise ValueError(msg.format(self.lamda))
