from moon_lander_3d import MoonLander3D
from ddpg_trainer import DDPGAgent, TrainingManager

env = MoonLander3D()
agent = DDPGAgent(state_dim=env.observation_dim, action_dim=env.action_dim)
trainer = TrainingManager(env, agent)
trainer.train(num_episodes=2000)