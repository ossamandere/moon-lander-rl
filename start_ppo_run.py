from moon_lander_3d import MoonLander3D
from ppo_trainer import PPOAgent
from ppo_trainer import PPOTrainingManager

env = MoonLander3D()

# Defaults in case we want to go back
default_state_dim = env.observation_dim
default_action_dim = env.action_dim
default_learning_rate = 3e-4
default_gamma = 0.2
default_lambda_gae = 0.95
default_clip_epsilon = 0.2
default_value_coeff = 0.5
default_entropy_coeff = 0.01
default_max_grad_norm = 0.5
default_hidden_size = 256

state_dim = default_state_dim
action_dim = default_action_dim
learning_rate = default_learning_rate
gamma = default_gamma
lambda_gae = default_lambda_gae
clip_epsilon = default_clip_epsilon
value_coeff = default_value_coeff
entropy_coeff = default_entropy_coeff # Change to 0.05 for exploration forcing
max_grad_norm = default_max_grad_norm
hidden_size = default_hidden_size
ppo_agent = PPOAgent(state_dim, action_dim, learning_rate, gamma, lambda_gae,
                     clip_epsilon, value_coeff, entropy_coeff, max_grad_norm,
                     hidden_size)
ppo_trainer = PPOTrainingManager(env, ppo_agent)
ppo_trainer.train(num_iterations=100)  # Each iteration = ~2048 steps