import torch
import torch.nn as nn

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
