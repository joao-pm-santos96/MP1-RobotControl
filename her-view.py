import gym
import os

from stable_baselines3 import *

MODEL = DDPG

model_name = str(MODEL.__name__)
models_folder = f'./models/{model_name}'
logs_folder = f'./logs/{model_name}'

if not os.path.exists(models_folder):
    os.makedirs(models_folder)

if not os.path.exists(logs_folder):
    os.makedirs(logs_folder)

# Init env
env = gym.make('FetchReach-v1')
n_episodes = 25

# Init model
model = TD3.load(f'{models_folder}/9', env=env)

for _ in range(n_episodes):
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

env.close()
