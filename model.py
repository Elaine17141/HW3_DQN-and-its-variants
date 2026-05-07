import torch
import torch.nn as nn
import math

class DQN(nn.Module):
    def __init__(self, input_dim=64, hidden1=150, hidden2=100, output_dim=4):
        """
        Standard Naive DQN Network architecture as required for HW3-1.
        """
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )
        
    def forward(self, x):
        return self.model(x)

class DuelingDQN(nn.Module):
    def __init__(self, input_dim=64, hidden1=150, hidden2=100, output_dim=4):
        """
        Dueling DQN Network architecture as required for HW3-3.
        
        Concept:
        Standard DQN evaluates the Q-value Q(s, a) directly. However, in many states, 
        the choice of action does not strictly affect the state's inherent value.
        Dueling DQN splits the network into two streams:
        1. Value Stream (V(s)): Evaluates how good it is to be in state 's'.
        2. Advantage Stream (A(s, a)): Evaluates the relative advantage of taking action 'a' in state 's'.
        
        By decoupling V and A, the model learns state values more efficiently without needing
        to experience every action in that state, leading to faster and more stable convergence.
        """
        super(DuelingDQN, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )
        
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden2, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        
        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden2, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine Value and Advantage streams
        # Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
        # The mean subtraction ensures that the advantage function has zero mean,
        # which acts as a regularizer and addresses the issue of unidentifiability
        # between V(s) and A(s, a).
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for Rainbow DQN (Factorized Gaussian Noise).
    Replaces standard epsilon-greedy exploration.
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return nn.functional.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return nn.functional.linear(x, self.weight_mu, self.bias_mu)

class RainbowDQN(nn.Module):
    def __init__(self, input_dim=64, hidden1=150, hidden2=100, output_dim=4):
        """
        Rainbow DQN Network architecture (Double + Dueling + Noisy Nets).
        """
        super(RainbowDQN, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden2, hidden2),
            nn.ReLU(),
            NoisyLinear(hidden2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden2, hidden2),
            nn.ReLU(),
            NoisyLinear(hidden2, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

    def reset_noise(self):
        for name, module in self.named_modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

