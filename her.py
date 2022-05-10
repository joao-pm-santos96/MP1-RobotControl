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
import argparse

from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
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
ALG = TD3

N_TIMESTEPS = int(1e6)
REWARD_THRESHOLD=1

"""
CLASS DEFINITIONS
"""

"""
FUNCTIONS DEFINITIONS
"""
def train(env, param_file, models_folder, logs_folder):

    # Load hyper-parameters
    with open(param_file) as file:
        hyperparams = yaml.load(file, Loader=yaml.FullLoader)

    # Init model
    buffer_class = HerReplayBuffer
    buffer_params = hyperparams['buffer_params']
    policy_params = hyperparams['policy_params']

    model = ALG(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=hyperparams['learning_rate'],
        buffer_size=hyperparams['buffer_size'],
        learning_starts=hyperparams['learning_starts'],
        batch_size=hyperparams['batch_size'],
        tau=hyperparams['tau'],
        gamma=hyperparams['gamma'],
        target_policy_noise=hyperparams['target_policy_noise'],
        target_noise_clip=hyperparams['target_noise_clip'],
        train_freq=(1, 'episode'),
        policy_delay=2,
        gradient_steps=-1,
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
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=1)
    # eval_callback = EvalCallback(eval_env, eval_freq=1000, callback_after_eval=callback_on_best, verbose=1)

    # Train the model
    model.learn(total_timesteps=N_TIMESTEPS, reset_num_timesteps=False, callback=[checkpoint_callback])

    # Save the best model
    model.save(f'{models_folder}/final')

def view(env, model, n_episodes):

    # Init model
    model = ALG.load(model, env=env)

    for _ in range(n_episodes):
        obs = env.reset()
        done = False

        while not done:
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-env', '--environment', help='Environment name', choices=['FetchPickAndPlace-v1', 'FetchPush-v1', 'FetchReach-v1', 'FetchSlide-v1'], required=True)
    parser.add_argument('-t', '--train', help='Train the model', action='store_true')
    parser.add_argument('-m', '--model', help='Path to model file to view')

    args = parser.parse_args()

    alg_name = str(ALG.__name__)
    models_folder = f'./models/{args.environment}/{alg_name}'
    logs_folder = f'./logs/{args.environment}/{alg_name}'

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    # Init env
    env = gym.make(args.environment)

    if args.train:
        train(env, f'./{args.environment}.yaml', models_folder, logs_folder)
    else:
        view(env, args.model, 25)

    env.close()

"""
MAIN
"""
if __name__ == '__main__':
    main()