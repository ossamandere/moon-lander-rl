import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import collections
from typing import Tuple, List, Dict, Any, Optional
import pickle
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PPOMemory:
    """Memory buffer for PPO algorithm"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def store(self, state: np.ndarray, action: np.ndarray, log_prob: float, 
              value: float, reward: float, done: bool):
        """Store experience"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_advantages(self, gamma: float = 0.99, lambda_gae: float = 0.95):
        """Compute advantages using Generalized Advantage Estimation (GAE)"""
        advantages = []
        returns = []
        
        # Convert to numpy arrays for easier computation
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Compute returns and advantages
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 1 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1 - dones[t]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            gae = delta + gamma * lambda_gae * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batches(self, batch_size: int):
        """Get mini-batches for training"""
        n_samples = len(self.states)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            
            yield {
                'states': torch.FloatTensor([self.states[i] for i in batch_indices]).to(device),
                'actions': torch.FloatTensor([self.actions[i] for i in batch_indices]).to(device),
                'old_log_probs': torch.FloatTensor([self.log_probs[i] for i in batch_indices]).to(device),
                'values': torch.FloatTensor([self.values[i] for i in batch_indices]).to(device),
                'advantages': torch.FloatTensor([self.advantages[i] for i in batch_indices]).to(device),
                'returns': torch.FloatTensor([self.returns[i] for i in batch_indices]).to(device)
            }
    
    def clear(self):
        """Clear all stored experiences"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()
    
    def __len__(self):
        return len(self.states)


class ActorCritic(nn.Module):
    """Combined Actor-Critic network for PPO"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super(ActorCritic, self).__init__()
        
        self.action_dim = action_dim
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head - outputs mean of actions
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
        # Actor head - outputs log standard deviation of actions
        #  self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * 0.5)  # Higher initial std
        
        # Critic head - outputs state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution and value"""
        features = self.shared_layers(state)
        
        # Actor output
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std.expand_as(action_mean))
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action_and_value(self, state: torch.Tensor, action: Optional[torch.Tensor] = None):
        """Get action, log probability, and value"""
        action_mean, action_std, value = self.forward(state)
        
        # Create action distribution
        dist = Normal(action_mean, action_std)
        
        if action is None:
            # Sample action
            action = dist.sample()
        
        # Apply bounds for moon lander actions
        bounded_action = self._apply_action_bounds(action)
        
        # Calculate log probability using the original (unbounded) action
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Calculate entropy for regularization
        entropy = dist.entropy().sum(dim=-1)
        
        return bounded_action, log_prob, entropy, value.squeeze(-1)
    
    def _apply_action_bounds(self, action: torch.Tensor) -> torch.Tensor:
        """Apply appropriate bounds for moon lander actions"""
        bounded_action = action.clone()
        
        # Thrust magnitude: sigmoid to [0, 1]
        # bounded_action[:, 0] = torch.sigmoid(action[:, 0])
        bounded_action[:, 0] = torch.clamp(action[:, 0], 0.0, 1.0)  # Direct clamping
        bounded_action[:, 1:] = torch.clamp(action[:, 1:], -1.0, 1.0)  # Direct clamping
        
        # Gimbal angles: tanh to [-1, 1]
        bounded_action[:, 1:] = torch.tanh(action[:, 1:])
        
        return bounded_action
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get state value"""
        features = self.shared_layers(state)
        return self.critic(features).squeeze(-1)


class PPOAgent:
    """PPO Agent for continuous control"""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, lambda_gae: float = 0.95, 
                 clip_epsilon: float = 0.2, value_coeff: float = 0.5,
                 entropy_coeff: float = 0.01, max_grad_norm: float = 0.5,
                 hidden_size: int = 256):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        
        # Network
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_size).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Memory
        self.memory = PPOMemory()
        
        # Training statistics
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.total_losses = []
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        self.actor_critic.eval()
        with torch.no_grad():
            if deterministic:
                # Use mean action for evaluation
                action_mean, _, value = self.actor_critic(state_tensor)
                bounded_action = self.actor_critic._apply_action_bounds(action_mean)
                action = bounded_action.cpu().numpy()[0]
                log_prob = 0.0  # Not used in deterministic mode
            else:
                # Sample action for training
                action, log_prob, _, value = self.actor_critic.get_action_and_value(state_tensor)
                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()
            
            value = value.cpu().numpy()
        
        self.actor_critic.train()

        # Debug what actions are being chosen
        if not deterministic and np.random.random() < 0.01:  # Debug 1% of actions
            print(f"Raw action: {action}")
            print(f"Action std: {self.actor_critic.actor_log_std.exp().detach().cpu().numpy()}")
            print(f"Thrust: {action[0]:.3f}, Gimbal: [{action[1]:.3f}, {action[2]:.3f}]")
            
        return action, log_prob, value
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, 
                        log_prob: float, value: float, reward: float, done: bool):
        """Store experience in memory"""
        self.memory.store(state, action, log_prob, value, reward, done)
    
    def update(self, epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """Update policy using PPO algorithm"""
        if len(self.memory) == 0:
            return {}
        
        # Compute advantages and returns
        self.memory.compute_advantages(self.gamma, self.lambda_gae)
        
        # Normalize advantages
        advantages = torch.FloatTensor(self.memory.advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.memory.advantages = advantages.tolist()
        
        # Training loop
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(epochs):
            for batch in self.memory.get_batches(batch_size):
                # Get current policy predictions
                _, log_probs, entropy, values = self.actor_critic.get_action_and_value(
                    batch['states'], batch['actions']
                )
                
                # Policy loss with clipping
                ratio = torch.exp(log_probs - batch['old_log_probs'])
                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch['advantages']
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch['returns'])
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_coeff * value_loss + 
                             self.entropy_coeff * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Clear memory
        self.memory.clear()
        
        # Store training statistics
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy_loss = np.mean(entropy_losses)
        avg_total_loss = avg_policy_loss + self.value_coeff * avg_value_loss + self.entropy_coeff * avg_entropy_loss
        
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropy_losses.append(avg_entropy_loss)
        self.total_losses.append(avg_total_loss)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_total_loss
        }
    
    def save(self, filepath: str):
        """Save agent state"""
        checkpoint = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
            'total_losses': self.total_losses,
            'hyperparameters': {
                'gamma': self.gamma,
                'lambda_gae': self.lambda_gae,
                'clip_epsilon': self.clip_epsilon,
                'value_coeff': self.value_coeff,
                'entropy_coeff': self.entropy_coeff,
                'max_grad_norm': self.max_grad_norm
            }
        }
        torch.save(checkpoint, filepath)
        print(f"PPO Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state"""
        try:
            # Try loading with weights_only=True (newer PyTorch default)
            checkpoint = torch.load(filepath, map_location=device, weights_only=True)
        except:
            # Fallback to weights_only=False for older checkpoints
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_losses = checkpoint.get('policy_losses', [])
        self.value_losses = checkpoint.get('value_losses', [])
        self.entropy_losses = checkpoint.get('entropy_losses', [])
        self.total_losses = checkpoint.get('total_losses', [])
        
        # Load hyperparameters if available
        if 'hyperparameters' in checkpoint:
            hp = checkpoint['hyperparameters']
            self.gamma = hp['gamma']
            self.lambda_gae = hp['lambda_gae']
            self.clip_epsilon = hp['clip_epsilon']
            self.value_coeff = hp['value_coeff']
            self.entropy_coeff = hp['entropy_coeff']
            self.max_grad_norm = hp['max_grad_norm']
        
        print(f"PPO Agent loaded from {filepath}")


class PPOTrainingManager:
    """Manages PPO training process"""
    
    def __init__(self, env, agent: PPOAgent, rollout_steps: int = 2048):
        self.env = env
        self.agent = agent
        self.rollout_steps = rollout_steps
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        self.training_stats = []

        # track recent performance
        self.reward_window = collections.deque(maxlen=50)
    
    def collect_rollout(self) -> List[Dict[str, Any]]:
        """Collect rollout data"""
        episodes_data = []
        steps_collected = 0
        
        while steps_collected < self.rollout_steps:
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_data = {'states': [], 'actions': [], 'rewards': [], 'done': False}
            
            while steps_collected < self.rollout_steps:
                # force early exploration
                # if steps_collected < 1000:  # First 1000 steps
                #     if np.random.random() < 0.3:  # 30% random actions
                #         action = np.array([
                #             np.random.random() * 0.8,  # Random thrust
                #             (np.random.random() - 0.5) * 0.4,  # Random gimbal
                #             (np.random.random() - 0.5) * 0.4
                #         ])
                #         log_prob, value = 0.0, 0.0  # Dummy values for exploration
                #     else:
                #         action, log_prob, value = self.agent.select_action(state, deterministic=False)
                # else:
                #     action, log_prob, value = self.agent.select_action(state, deterministic=False)

                # Select action
                action, log_prob, value = self.agent.select_action(state, deterministic=False)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.agent.store_experience(state, action, log_prob, value, reward, done)
                
                episode_reward += reward
                episode_length += 1
                steps_collected += 1
                
                state = next_state
                
                if done:
                    episodes_data.append({
                        'episode_reward': episode_reward,
                        'episode_length': episode_length,
                        'landing_result': info.get('landing_result', 'Unknown')
                    })
                    break
        
        return episodes_data
    
    def train(self, num_iterations: int, save_interval: int = 50, 
              eval_interval: int = 25, save_dir: str = "ppo_checkpoints"):
        """Main training loop"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        best_avg_reward = float('-inf')
        recent_successes = collections.deque(maxlen=100)
        iteration = 0
        
        print("Starting PPO training...")
        print(f"Iterations: {num_iterations}")
        print(f"Rollout steps per iteration: {self.rollout_steps}")
        print(f"State dim: {self.agent.state_dim}")
        print(f"Action dim: {self.agent.action_dim}")
        print("-" * 60)
        
        while iteration < num_iterations:
            # Collect rollout
            episodes_data = self.collect_rollout()
            
            # Update policy
            losses = self.agent.update()
            
            # Process episode statistics
            for episode_data in episodes_data:
                self.episode_rewards.append(episode_data['episode_reward'])
                self.episode_lengths.append(episode_data['episode_length'])
                
                # Track success rate
                is_success = episode_data['landing_result'] == 'SUCCESS'
                recent_successes.append(is_success)
            
            # Calculate statistics
            current_success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0
            recent_rewards = self.episode_rewards[-len(episodes_data):] if episodes_data else [0]
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean([ep['episode_length'] for ep in episodes_data]) if episodes_data else 0
            
            # Print progress
            if iteration % 5 == 0:
                grad_norm = losses.get('gradient_norm', 0)
                print(f"Iter {iteration:4d} | "
                      f"Episodes: {len(episodes_data):2d} | "
                      f"Avg Reward: {avg_reward:7.1f} | "
                      f"Avg Length: {avg_length:5.1f} | "
                      f"Success%: {current_success_rate:5.1%} | "
                      f"Policy Loss: {losses.get('policy_loss', 0):6.3f} | "
                      f"Grad Norm: {grad_norm:.4f}")

                # learning rate diagnostic
                if grad_norm < 1e-4:
                    print("WARNING: very small gradient norms - increase learning rate")
            
            # Evaluation and saving
            if iteration % eval_interval == 0 and iteration > 0:
                eval_results = self.evaluate(num_episodes=5, render=False)
                avg_eval_reward = eval_results['avg_reward']
                
                if avg_eval_reward > best_avg_reward:
                    best_avg_reward = avg_eval_reward
                    self.agent.save(os.path.join(save_dir, "best_ppo_agent.pth"))
                
                print(f"Evaluation - Avg Reward: {avg_eval_reward:.1f}, "
                      f"Success Rate: {eval_results['success_rate']:.1%}")
                
            # Performance Stagnation Check
            self.reward_window.extend(recent_rewards)

            if len(self.reward_window) == 50 and iteration > 100:
                early_rewards = list(self.reward_window)[:25]
                late_rewards = list(self.reward_window)[25:]
                improvement = np.mean(late_rewards) - np.mean(early_rewards)
                if abs(improvement) < 10: # may need adjusting
                    print(f"WARNING: Minimal improvement ({improvement:.1f}), learning rate too low")
                
            
            if iteration % save_interval == 0 and iteration > 0:
                self.agent.save(os.path.join(save_dir, f"ppo_agent_iter_{iteration}.pth"))
                self.save_training_stats(os.path.join(save_dir, f"ppo_training_stats_{iteration}.pkl"))
            
            iteration += 1
        
        print("PPO Training completed!")
        self.agent.save(os.path.join(save_dir, "final_ppo_agent.pth"))
        self.save_training_stats(os.path.join(save_dir, "final_ppo_training_stats.pkl"))

        # Monitors policy loss trends to check learning rate
        if iteration > 0 and iteration % 10 == 0:
            recent_policy_losses = self.agent.policy_losses[-10:]
            policy_loss_change = abs(recent_policy_losses[-1] - recent_policy_losses[0])
            print(f"Policy loss change (last 10 iters): {policy_loss_change:.6f}")
            
            # Flag if change is too small
            if policy_loss_change < 1e-5:
                print("WARNING: Policy loss barely changing - learning rate might be too low")
    
    def evaluate(self, num_episodes: int = 10, render: bool = True) -> Dict[str, Any]:
        """Evaluate the trained agent"""
        eval_rewards = []
        eval_lengths = []
        success_count = 0
        
        print(f"Evaluating PPO agent for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Select action deterministically
                action, _, _ = self.agent.select_action(state, deterministic=True)
                state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if render and episode < 3:
                    self.env.render()
                
                if done:
                    if info.get('landing_result') == 'SUCCESS':
                        success_count += 1
                    
                    if render:
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
        
        if render:
            print(f"\nPPO Evaluation Results:")
            print(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"Success Rate: {results['success_rate']:.1%}")
            print(f"Average Episode Length: {results['avg_length']:.1f}")
        
        return results
    
    def save_training_stats(self, filepath: str):
        """Save training statistics"""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.agent.policy_losses,
            'value_losses': self.agent.value_losses,
            'entropy_losses': self.agent.entropy_losses,
            'total_losses': self.agent.total_losses
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(stats, f)


# Example usage
if __name__ == "__main__":
    # Import the moon lander environment (assumes it's in the same directory)
    # from moon_lander_env import MoonLander3D
    
    # Uncomment and run this section when you have the environment file:
    """
    # Create environment and agent
    env = MoonLander3D()
    agent = PPOAgent(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        lr=3e-4,
        clip_epsilon=0.2
    )
    
    # Create training manager
    trainer = PPOTrainingManager(env, agent, rollout_steps=2048)
    
    # Train the agent
    trainer.train(num_iterations=500, save_interval=50, eval_interval=25)
    
    # Evaluate the trained agent
    trainer.evaluate(num_episodes=10, render=True)
    """
    
    print("PPO Training code ready!")
    print("To use: Import MoonLander3D, then uncomment and run the example usage section.")