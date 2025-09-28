import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import collections
from typing import Tuple, List, Dict, Any
import pickle
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ReplayBuffer:
    """Experience replay buffer for DDPG"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch]).to(device)
        actions = torch.FloatTensor([e[1] for e in batch]).to(device)
        rewards = torch.FloatTensor([e[2] for e in batch]).unsqueeze(1).to(device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
        dones = torch.BoolTensor([e[4] for e in batch]).unsqueeze(1).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network for DDPG - outputs continuous actions"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super(Actor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass - outputs raw actions"""
        return self.network(state)
    
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get bounded actions for moon lander"""
        raw_actions = self.forward(state)
        
        # Apply appropriate activation functions for each action component
        # thrust_magnitude: 0 to 1 (sigmoid)
        # gimbal_x, gimbal_y: -1 to 1 (tanh)
        bounded_actions = torch.cat([
            torch.sigmoid(raw_actions[:, :1]),  # thrust_magnitude [0, 1]
            torch.tanh(raw_actions[:, 1:])      # gimbal angles [-1, 1]
        ], dim=1)
        
        return bounded_actions


class Critic(nn.Module):
    """Critic network for DDPG - estimates Q-values"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super(Critic, self).__init__()
        
        # State processing layers
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU()
        )
        
        # Combined state-action processing
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_size + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass - outputs Q-value"""
        state_features = self.state_net(state)
        combined = torch.cat([state_features, action], dim=1)
        q_value = self.combined_net(combined)
        return q_value


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for action exploration"""
    
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, 
                 sigma: float = 0.2, dt: float = 1e-2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()
    
    def reset(self):
        """Reset the noise process"""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self) -> np.ndarray:
        """Sample noise"""
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.state += dx
        return self.state


class DDPGAgent:
    """DDPG Agent for continuous control"""
    
    def __init__(self, state_dim: int, action_dim: int, lr_actor: float = 1e-4, 
                 lr_critic: float = 1e-3, gamma: float = 0.99, tau: float = 0.005,
                 hidden_size: int = 256):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_size).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_size).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_size).to(device)
        
        # Copy parameters to target networks
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Noise for exploration
        self.noise = OrnsteinUhlenbeckNoise(action_dim)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training statistics
        self.actor_losses = []
        self.critic_losses = []
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.get_action(state_tensor).cpu().numpy()[0]
        self.actor.train()
        
        if add_noise:
            noise = self.noise.sample()
            # Scale noise appropriately for each action component
            noise[0] *= 0.1   # Smaller noise for thrust (0-1 range)
            noise[1:] *= 0.2  # Moderate noise for gimbal (-1 to 1 range)
            action += noise
            
            # Clip to valid ranges
            action[0] = np.clip(action[0], 0.0, 1.0)    # thrust
            action[1:] = np.clip(action[1:], -1.0, 1.0) # gimbal
        
        return action
    
    def train(self, batch_size: int = 128) -> Dict[str, float]:
        """Train the agent"""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Train Critic
        with torch.no_grad():
            next_actions = self.actor_target.get_action(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (self.gamma * target_q * ~dones)
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Train Actor
        predicted_actions = self.actor.get_action(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)
        
        # Store losses
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }
    
    def soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """Soft update target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def hard_update(self, target: nn.Module, source: nn.Module):
        """Hard update target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save(self, filepath: str):
        """Save agent state"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses
        }
        torch.save(checkpoint, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_losses = checkpoint.get('actor_losses', [])
        self.critic_losses = checkpoint.get('critic_losses', [])
        
        print(f"Agent loaded from {filepath}")


class TrainingManager:
    """Manages the training process"""
    
    def __init__(self, env, agent: DDPGAgent):
        self.env = env
        self.agent = agent
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        self.training_stats = []
    
    def train_episode(self) -> Dict[str, Any]:
        """Train for one episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        training_losses = []
        
        self.agent.noise.reset()  # Reset noise for new episode
        
        while True:
            # Select action
            action = self.agent.select_action(state, add_noise=True)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train agent
            if len(self.agent.replay_buffer) > 1000:  # Start training after some experiences
                losses = self.agent.train()
                if losses:
                    training_losses.append(losses)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'landing_result': info.get('landing_result', 'Unknown'),
            'training_losses': training_losses
        }
    
    def train(self, num_episodes: int, save_interval: int = 100, 
              eval_interval: int = 50, save_dir: str = "checkpoints"):
        """Main training loop"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        best_avg_reward = float('-inf')
        recent_successes = collections.deque(maxlen=100)  # Track success rate
        
        print("Starting training...")
        print(f"Episodes: {num_episodes}")
        print(f"State dim: {self.agent.state_dim}")
        print(f"Action dim: {self.agent.action_dim}")
        print("-" * 50)
        
        for episode in range(num_episodes):
            # Train episode
            episode_stats = self.train_episode()
            
            # Track statistics
            self.episode_rewards.append(episode_stats['episode_reward'])
            self.episode_lengths.append(episode_stats['episode_length'])
            
            # Track success rate
            is_success = episode_stats['landing_result'] == 'SUCCESS'
            recent_successes.append(is_success)
            current_success_rate = sum(recent_successes) / len(recent_successes)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_stats['episode_reward']:7.1f} | "
                      f"Avg10: {avg_reward:7.1f} | "
                      f"Length: {episode_stats['episode_length']:3d} | "
                      f"Success%: {current_success_rate:5.1%} | "
                      f"Result: {episode_stats['landing_result']}")
            
            # Evaluation and saving
            if episode % eval_interval == 0 and episode > 0:
                avg_reward = np.mean(self.episode_rewards[-eval_interval:])
                
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.agent.save(os.path.join(save_dir, "best_agent.pth"))
            
            if episode % save_interval == 0 and episode > 0:
                self.agent.save(os.path.join(save_dir, f"agent_episode_{episode}.pth"))
                self.save_training_stats(os.path.join(save_dir, f"training_stats_{episode}.pkl"))
        
        print("Training completed!")
        self.agent.save(os.path.join(save_dir, "final_agent.pth"))
        self.save_training_stats(os.path.join(save_dir, "final_training_stats.pkl"))
    
    def evaluate(self, num_episodes: int = 10, render: bool = True) -> Dict[str, Any]:
        """Evaluate the trained agent"""
        eval_rewards = []
        eval_lengths = []
        success_count = 0
        
        print(f"Evaluating agent for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Select action without noise
                action = self.agent.select_action(state, add_noise=False)
                state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if render and episode < 3:  # Render first few episodes
                    self.env.render()
                
                if done:
                    if info.get('landing_result') == 'SUCCESS':
                        success_count += 1
                    
                    print(f"Eval Episode {episode}: Reward={episode_reward:.1f}, "
                          f"Result={info.get('landing_result', 'Unknown')}")
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        results = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_length': np.mean(eval_lengths),
            'success_rate': success_count / num_episodes,
            'episodes': eval_rewards
        }
        
        print(f"\nEvaluation Results:")
        print(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Average Episode Length: {results['avg_length']:.1f}")
        
        return results
    
    def save_training_stats(self, filepath: str):
        """Save training statistics"""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'actor_losses': self.agent.actor_losses,
            'critic_losses': self.agent.critic_losses
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(stats, f)


# Example usage
if __name__ == "__main__":
    # Import the moon lander environment (assumes it's in the same directory)
    # from moon_lander_env import MoonLander3D
    
    # Uncomment and run this section when you have both files:
    """
    # Create environment and agent
    env = MoonLander3D()
    agent = DDPGAgent(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        lr_actor=1e-4,
        lr_critic=1e-3
    )
    
    # Create training manager
    trainer = TrainingManager(env, agent)
    
    # Train the agent
    trainer.train(num_episodes=2000, save_interval=200, eval_interval=100)
    
    # Evaluate the trained agent
    trainer.evaluate(num_episodes=10, render=True)
    """
    
    print("DDPG Training code ready!")
    print("To use: Import MoonLander3D, then uncomment and run the example usage section.")