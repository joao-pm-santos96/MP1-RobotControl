import gym
import os

from stable_baselines3 import *

MODEL = DDPG
N_TIMESTEPS = 10000
EPOCHS = 100

model_name = str(MODEL.__name__)
models_folder = f'./models/{model_name}/'
logs_folder = f'./logs/{model_name}/'

if not os.path.exists(models_folder):
    os.makedirs(models_folder)

if not os.path.exists(logs_folder):
    os.makedirs(logs_folder)

# Init env
env = gym.make('FetchReach-v1')

# Init model
model = MODEL(
    "MultiInputPolicy",
    env,
    replay_buffer_class = HerReplayBuffer,
    learning_rate = 0.1,
    tau = 0.005,
    gamma = 0.99,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
        online_sampling=True
    ),
    verbose=1,
    tensorboard_log=logs_folder
)

for i in range(EPOCHS):
    # Train the model
    model.learn(total_timesteps=N_TIMESTEPS, reset_num_timesteps=False)

    # Save the model
    model.save(f'{models_folder}/{i}')







