import pytorch_lightning as pl
from torch.utils.data import DataLoader
from agent import GridWorldAgent, DummyDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np

def train(mode='static', algo='double_dueling_dqn', steps=50000):
    # Instantiate the LightningModule
    agent = GridWorldAgent(mode=mode, algo=algo, epsilon_decay_steps=steps)
    
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
    
    print(f"Starting PyTorch Lightning training with '{algo}' in '{mode}' mode for {steps} steps...")
    trainer.fit(agent, dataloader)
    print("Training finished!")
    
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='static', choices=['static', 'player', 'random'],
                        help="Mode for GridWorld: static, player, or random")
    parser.add_argument('--algo', type=str, default='double_dueling_dqn', 
                        choices=['dqn', 'double_dqn', 'dueling_dqn', 'double_dueling_dqn', 'rainbow'],
                        help="DQN Algorithm variant to use")
    parser.add_argument('--steps', type=int, default=20000, help="Number of environmental steps to train")
    args = parser.parse_args()
    
    agent = train(mode=args.mode, algo=args.algo, steps=args.steps)
    
    # Save the model
    model_path = f"model_{args.algo}_{args.mode}_pl.pth"
    torch.save(agent.main_net.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    if args.algo == 'double_dueling_dqn':
        title_algo = 'Dueling Double DQN'
    elif args.algo == 'dueling_dqn':
        title_algo = 'Dueling DQN'
    elif args.algo == 'double_dqn':
        title_algo = 'Double DQN'
    elif args.algo == 'rainbow':
        title_algo = 'Rainbow DQN'
    else:
        title_algo = 'Naive DQN'

    # Plot rewards
    epoch_rewards = agent.episode_rewards
    plt.figure(figsize=(10, 7))
    plt.plot(epoch_rewards)
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Total Reward", fontsize=11)
    plt.title(f"{title_algo} Training Reward", fontsize=13)
        
    plot_path = f"reward_plot_{args.algo}_{args.mode}_pl.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Reward plot saved to {plot_path}")
    
    # Plot losses
    if hasattr(agent, 'episode_losses') and len(agent.episode_losses) > 0:
        epoch_losses = agent.episode_losses
        plt.figure(figsize=(10, 7))
        plt.plot(epoch_losses)
        plt.xlabel("Epochs", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.title(f"{title_algo} Training Loss", fontsize=13)
            
        loss_plot_path = f"loss_plot_{args.algo}_{args.mode}_pl.png"
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Loss plot saved to {loss_plot_path}")
