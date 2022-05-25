from unittest.mock import NonCallableMagicMock
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    """General MLP"""

    def __init__(self, in_size: int, out_size: int, hidden_sizes: list, seed: int=0, out_gate=None):
        """Initialize MLP with given hidden layer sizes
        Params
        ======
            in_size: Dimension of input
            out_size: Dimension of output
            hidden_sizes: List of hidden layer sizes
            seed: Random seed
            out_gate: Output gate
        """
        super(Network, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.out_gate = out_gate
        self.linear_layers = nn.ModuleList()

        last_size = in_size
        for size in hidden_sizes:
            self.linear_layers.append(nn.Linear(last_size, size))
            last_size = size

        self.linear_layers.append(nn.Linear(last_size, out_size))

        # Ref: https://discuss.pytorch.org/t/understanding-enropy/34677
        self.std = nn.Parameter(torch.ones(1, out_size))    # trainable std
    
    def forward(self, input: torch.Tensor, add_noise: bool=False, min=-1, max=1):
        """Forward pass of the network"""
        x = input
        for layer in self.linear_layers[:-1]:
            x = F.relu(layer(x))
        
        # Last layer
        x = self.linear_layers[-1](x)

        # Apply output gate if exist
        if self.out_gate:
            x = self.out_gate(x)

        if add_noise:
            dist = torch.distributions.Normal(x, F.softplus(self.std))
            x = torch.clamp(dist.rsample(), min=min, max=max)

        return x

# class ActorNet(nn.Module):
#     """Actor network with normal distributed noise"""
#     def __init__(self, state_size: int, action_size: int, hidden_sizes: list, seed: int=0, out_gate=None):
#         super(ActorNet, self).__init__()

#         self.network_body = Network(state_size, action_size, hidden_sizes, seed, out_gate)

#         # Ref: https://discuss.pytorch.org/t/understanding-enropy/34677
#         self.std = nn.Parameter(torch.ones(1, action_size))    # trainable std

#     def forward(self, state: torch.Tensor):
#         action = self.network_body(state)
#         dist = torch.distributions.Normal(action, self.std)
#         action = torch.clamp(dist.sample(), min=-1, max=1)
#         return action