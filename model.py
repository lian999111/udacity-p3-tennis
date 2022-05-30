import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorNet(nn.Module):
    """Actor network with normal distributed noise"""
    def __init__(self, state_size: int, action_size: int, hidden_sizes: list, seed: int=0):
        super(ActorNet, self).__init__()

        self.seed = torch.manual_seed(seed)
        
        self.linear_layers = nn.ModuleList()
        self.out_gate = F.tanh

        last_size = state_size
        for size in hidden_sizes:
            self.linear_layers.append(nn.Linear(last_size, size))
            last_size = size

        self.linear_layers.append(nn.Linear(last_size, action_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Heurisitic initialization of fully connected layers"""
        for layer in self.linear_layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))

        self.linear_layers[-1].weight.data.uniform_(-3e-3, 3e-3)
        # self.linear_layers[-1].bias.data.fill_(0.1)

    def forward(self, input: torch.Tensor, min=-1, max=1):
        """Forward pass of the network"""
        x = input
        for layer in self.linear_layers[:-1]:
            x = F.relu(layer(x))
        
        # Last layer
        x = self.linear_layers[-1](x)
        x = self.out_gate(x)

        return x

class CriticNet(nn.Module):
    """Critic network with normal distributed noise"""
    def __init__(self, state_size: int, action_size: int, hidden_sizes: list, seed: int=0):
        super(CriticNet, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, hidden_sizes[0])                       # only state input
        self.fc2 = nn.Linear(hidden_sizes[0]+action_size, hidden_sizes[1])      # action input fed here
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Heurisitic initialization of fully connected layers"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """Forward pass of the network"""
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)