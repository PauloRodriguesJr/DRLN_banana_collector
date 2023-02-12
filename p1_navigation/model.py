import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, num_neurons=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"

        self.fc1 = nn.Linear(state_size, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        self.fc_advantage = nn.Linear(num_neurons, action_size)
        self.fc_baseline = nn.Linear(num_neurons, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        advantage = F.relu(self.fc_advantage(x))
        baseline = F.relu(self.fc_baseline(x))

        return advantage, baseline
