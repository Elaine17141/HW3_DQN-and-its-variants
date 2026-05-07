import pytorch_lightning as pl
from torch.utils.data import DataLoader
from agent import DoubleDuelingDQNAgent, DummyDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np

def train(mode='static', steps=50000):
    # Instantiate the LightningModule
    agent = DoubleDuelingDQNAgent(mode=mode, epsilon_decay_steps=steps)
    
    # Create a dummy dataloader. One step in DataLoader = one step in environment
    dataset = DummyDataset(steps)
    dataloader = DataLoader(dataset, batch_size=1)
    
    # Configure the PyTorch Lightning Trainer
    # Add Gradient Clipping (gradient_clip_val=1.0) for stability
    trainer = pl.Trainer(
        max_epochs=1, 
        gradient_clip_val=1.0, 
        enable_progress_bar=True,
        logger=False, # Disable default logger to avoid clutter
        enable_checkpointing=False
    )
    
    print(f"Starting PyTorch Lightning training in '{mode}' mode for {steps} steps...")
    trainer.fit(agent, dataloader)
    print("Training finished!")
    
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='static', choices=['static', 'player', 'random'],
                        help="Mode for GridWorld: static, player, or random")
    parser.add_argument('--steps', type=int, default=20000, help="Number of environmental steps to train")
    args = parser.parse_args()
    
    agent = train(mode=args.mode, steps=args.steps)
    
    # Save the model
    model_path = f"dqn_model_{args.mode}_pl.pth"
    torch.save(agent.main_net.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot rewards
    epoch_rewards = agent.episode_rewards
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Reward over time ({args.mode} mode) [Double Dueling DQN]')
    
    # Optional: Plot moving average for better visualization
    window = 50
    if len(epoch_rewards) >= window:
        moving_avg = np.convolve(epoch_rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(epoch_rewards)), moving_avg, color='red', label='Moving Average')
        plt.legend()
        
    plot_path = f"reward_plot_{args.mode}_pl.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Reward plot saved to {plot_path}")
