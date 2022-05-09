#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
import gym
import os
import yaml
import numpy as np

from stable_baselines3 import HerReplayBuffer, SAC, DDPG
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold

"""
TODO
"""

"""
PARAMETERS
"""
TRAINING = True

ALG = DDPG

# ENV = 'FetchPickAndPlace-v1'
ENV = 'FetchPush-v1'
# ENV = 'FetchReach-v1'
# ENV = 'FetchSlide-v1'

N_TIMESTEPS = int(1e6)
REWARD_THRESHOLD=-1

"""
CLASS DEFINITIONS
"""

"""
FUNCTIONS DEFINITIONS
"""
def train(env, alg_name, models_folder, logs_folder):

    # Load hyper-parameters
    with open(r'./hyperparams.yaml') as file:
        hyperparams = yaml.load(file, Loader=yaml.FullLoader)[ENV]

    # Init eval env
    eval_env = Monitor(env)

    # Init model
    buffer_class = HerReplayBuffer
    buffer_params = hyperparams['buffer_params']
    policy_params = hyperparams['policy_params']

    # The noise objects for DDPG
    if alg_name == 'DDPG':
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
    else:
        action_noise = None

    model = ALG(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=hyperparams['learning_rate'],
        buffer_size=hyperparams['buffer_size'],
        learning_starts=hyperparams['learning_starts'],
        batch_size=hyperparams['batch_size'],
        tau=hyperparams['tau'],
        gamma=hyperparams['gamma'],
        train_freq=(1, 'episode'),
        gradient_steps=-1,
        # ent_coef='auto',
        action_noise=action_noise,
        replay_buffer_class = buffer_class,
        replay_buffer_kwargs= buffer_params,
        policy_kwargs=policy_params,
        optimize_memory_usage=False,
        create_eval_env=False,
        seed=None,
        tensorboard_log=logs_folder,
        verbose=1
    )

    # Callbacks
    # Save a checkpoint every x steps
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=models_folder, name_prefix='model')

    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=1)
    eval_callback = EvalCallback(eval_env, eval_freq=1000, callback_after_eval=callback_on_best, verbose=1)

    # Train the model
    model.learn(total_timesteps=N_TIMESTEPS, reset_num_timesteps=False, callback=[checkpoint_callback, eval_callback])

    # Save the best model
    model.save(f'{models_folder}/final_{ENV}')

def view(env, models_folder, n_episodes):

    # Init model
    model = ALG.load(f'{models_folder}/final_{ENV}', env=env)

    for _ in range(n_episodes):
        obs = env.reset()
        done = False

        while not done:
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)

def main():

    alg_name = str(ALG.__name__)
    models_folder = f'./models/{ENV}/{alg_name}/'
    logs_folder = f'./logs/{ENV}/{alg_name}/'

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    # Init env
    env = gym.make(ENV)

    if TRAINING:
        train(env, alg_name, models_folder, logs_folder)
    else:
        view(env, models_folder, 25)

    env.close()

"""
MAIN
"""
if __name__ == '__main__':
    main()