import gymnasium as gym
env = gym.make("CarRacing-v2", domain_randomize=True)

# normal reset, this changes the colour scheme by default
env.reset()
env.render()