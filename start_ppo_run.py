from moon_lander_3d import MoonLander3D
from ppo_trainer import PPOAgent
from ppo_trainer import PPOTrainingManager

env = MoonLander3D()
ppo_agent = PPOAgent(state_dim=env.observation_dim, action_dim=env.action_dim)
ppo_trainer = PPOTrainingManager(env, ppo_agent)
ppo_trainer.train(num_iterations=100)  # Each iteration = ~2048 steps