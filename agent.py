import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from model import DQN, DuelingDQN, RainbowDQN
from Gridworld import Gridworld

class ReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

def get_state(game):
    # Added random noise to avoid getting stuck in identical states as per Chapter 3
    state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    state = torch.from_numpy(state_).float()
    return state

class GridWorldAgent(pl.LightningModule):
    def __init__(self, mode='static', algo='double_dueling_dqn', lr=1e-3, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=10000, mem_size=1000, batch_size=200, target_update_freq=50):
        super().__init__()
        self.save_hyperparameters()
        self.mode = mode
        self.algo = algo.lower()
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        
        # Select Architecture
        if self.algo == 'rainbow':
            self.main_net = RainbowDQN()
            self.target_net = RainbowDQN()
        elif 'dueling' in self.algo:
            self.main_net = DuelingDQN()
            self.target_net = DuelingDQN()
        else:
            self.main_net = DQN()
            self.target_net = DQN()
            
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()
        
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer(maxlen=mem_size)
        
        # Environment setup
        self.env = Gridworld(size=4, mode=self.mode)
        self.state = get_state(self.env)
        self.episode_reward = 0
        self.episode_rewards = []
        self.episode_loss_sum = 0
        self.episode_loss_count = 0
        self.episode_losses = []
        self.moves = 0
        self.max_moves = 50
        
        # Pre-populate replay buffer
        self.populate_buffer(self.batch_size)

    def populate_buffer(self, steps):
        for _ in range(steps):
            action_idx = random.randint(0, 3)
            action = self.action_set[action_idx]
            self.env.makeMove(action)
            next_state = get_state(self.env)
            reward = self.env.reward()
            done = True if reward > 0 else False
            
            self.replay.add((self.state, action_idx, reward, next_state, done))
            self.state = next_state
            self.moves += 1
            
            if reward != -1 or self.moves >= self.max_moves:
                self.env = Gridworld(size=4, mode=self.mode)
                self.state = get_state(self.env)
                self.moves = 0

    def forward(self, x):
        return self.main_net(x)

    def get_action(self, state):
        if self.algo == 'rainbow':
            # Rainbow uses Noisy Nets for exploration, no epsilon-greedy
            with torch.no_grad():
                qval = self.main_net(state)
            return torch.argmax(qval, dim=1).item()
        else:
            if random.random() < self.epsilon:
                return random.randint(0, 3)
            else:
                with torch.no_grad():
                    qval = self.main_net(state)
                return torch.argmax(qval, dim=1).item()

    def training_step(self, batch, batch_idx):
        if self.algo == 'rainbow':
            # Reset noise before picking an action and calculating TD target
            self.main_net.reset_noise()
            self.target_net.reset_noise()
            
        # 1. Step the environment
        action_idx = self.get_action(self.state)
        action = self.action_set[action_idx]
        self.env.makeMove(action)
        next_state = get_state(self.env)
        reward = self.env.reward()
        done = True if reward > 0 else False
        
        self.replay.add((self.state, action_idx, reward, next_state, done))
        self.episode_reward += reward
        self.state = next_state
        self.moves += 1
        
        if reward != -1 or self.moves >= self.max_moves:
            self.episode_rewards.append(self.episode_reward)
            self.log('episode_reward', self.episode_reward, prog_bar=True)
            self.episode_reward = 0
            
            if getattr(self, 'episode_loss_count', 0) > 0:
                self.episode_losses.append(self.episode_loss_sum / self.episode_loss_count)
            else:
                self.episode_losses.append(0.0)
            self.episode_loss_sum = 0
            self.episode_loss_count = 0
            
            self.moves = 0
            self.env = Gridworld(size=4, mode=self.mode)
            self.state = get_state(self.env)
            
        # Decay epsilon (Not needed for Rainbow)
        if self.algo != 'rainbow' and self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_end)
            self.log('epsilon', self.epsilon, prog_bar=True)
            
        # 2. Sample from buffer and calculate loss
        minibatch = self.replay.sample(self.batch_size)
        state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
        action_batch = torch.tensor([a for (s1, a, r, s2, d) in minibatch], dtype=torch.long)
        reward_batch = torch.tensor([r for (s1, a, r, s2, d) in minibatch], dtype=torch.float32)
        state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
        done_batch = torch.tensor([d for (s1, a, r, s2, d) in minibatch], dtype=torch.float32)
        
        # Calculate Q values
        with torch.no_grad():
            if 'double' in self.algo or self.algo == 'rainbow':
                # Double DQN Logic
                next_actions = torch.argmax(self.main_net(state2_batch), dim=1).unsqueeze(1)
                next_q_values = self.target_net(state2_batch).gather(1, next_actions).squeeze(1)
            else:
                # Standard Naive DQN Logic
                next_q_values = torch.max(self.target_net(state2_batch), dim=1)[0]
        
        # Target Q value
        Y = reward_batch + self.gamma * ((1 - done_batch) * next_q_values)
        
        # Current Q value
        Q1 = self.main_net(state1_batch)
        X = Q1.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        loss = self.loss_fn(X, Y)
        self.log('train_loss', loss, prog_bar=True)
        
        self.episode_loss_sum += loss.item()
        self.episode_loss_count += 1
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update Target Network periodically
        if self.global_step > 0 and self.global_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

    def configure_optimizers(self):
        return torch.optim.Adam(self.main_net.parameters(), lr=self.lr)

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size): 
        self.size = size
    def __len__(self): 
        return self.size
    def __getitem__(self, idx): 
        return 0
