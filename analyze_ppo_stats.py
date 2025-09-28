import pickle
import numpy as np

# Load the training stats
with open('ppo_checkpoints/ppo_training_stats_450.pkl', 'rb') as f:
    stats = pickle.load(f)

# Check what's in there
print("Available data:", stats.keys())

# Look at recent episode results
recent_rewards = stats['episode_rewards'][-50:]  # Last 50 episodes
print(f"Recent rewards: {np.mean(recent_rewards):.1f} ± {np.std(recent_rewards):.1f}")
print(f"Best reward: {max(stats['episode_rewards']):.1f}")
print(f"Total episodes: {len(stats['episode_rewards'])}")