import gym
import os

from stable_baselines3 import *

MODEL = SAC
N_TIMESTEPS = int(5e5)
EPOCHS = 1

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
buffer_class = HerReplayBuffer
buffer_params = dict(
    n_sampled_goal=4,
    goal_selection_strategy='future',
    online_sampling=True
)

policy_params = dict(net_arch=[64, 64], n_critics=1)

model = MODEL(
    policy="MultiInputPolicy",
    env=env,
    learning_rate = 1e-3,
    buffer_size=int(1000000),
    learning_starts=1000,
    batch_size=256,
    tau=0.95,
    gamma=0.95,
    train_freq=(1, 'episode'),
    gradient_steps=-1,
    ent_coef='auto',
    action_noise=None,
    replay_buffer_class = buffer_class,
    replay_buffer_kwargs= buffer_params,
    policy_kwargs=policy_params,
    optimize_memory_usage=False,
    create_eval_env=False,
    seed=None,
    verbose=1
)

for i in range(EPOCHS):
    # Train the model
    model.learn(total_timesteps=N_TIMESTEPS, reset_num_timesteps=False)

    # Save the model
    model.save(f'{models_folder}/{i}')







