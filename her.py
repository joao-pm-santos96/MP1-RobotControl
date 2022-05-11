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

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold

from agents import Agents
"""
TODO
"""

"""
PARAMETERS
"""
REWARD_THRESHOLD=-5
TIMESTEPS=int(1e6)
ENVS = ['FetchReach-v1', 'FetchSlide-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1']

"""
CLASS DEFINITIONS
"""

"""
FUNCTIONS DEFINITIONS
"""
def train(env):

    env_name = env.unwrapped.spec.id
    agent = Agents(env_name)

    alg_name = agent.alg_name
    models_folder = f'./models/{env_name}/{alg_name}'
    logs_folder = f'./logs/{env_name}'

    model = agent.gen_model(env, logs_folder=logs_folder)

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    # Callbacks
    # Save a checkpoint every x steps
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=models_folder, name_prefix='model')

    # Stop training when the model reaches the reward threshold
    eval_env = Monitor(env)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=1)
    eval_callback = EvalCallback(eval_env, eval_freq=10000, callback_after_eval=callback_on_best, verbose=1)

    # Train the model
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=[checkpoint_callback, eval_callback])

    # Save the best model
    model.save(f'{models_folder}/final')

def view(env, model_file=None, n_episodes=25):

    # Init model
    if model_file:
        env_name = env.unwrapped.spec.id
        agent = Agents(env_name)
        model = agent.get_model(model_file, env)

    for _ in range(n_episodes):
        obs = env.reset()
        done = False

        while not done:
            env.render()
            
            if model_file:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, done, _ = env.step(action)

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-env', '--environment', help='Environment name', choices=ENVS, required=True)
    parser.add_argument('-t', '--train', help='Train the model', action='store_true')
    parser.add_argument('-m', '--model_file', help='Path to model file to view', default=None)

    args = parser.parse_args()    

    # Init env
    env = gym.make(args.environment)

    if args.train:
        train(env)
    else:
        view(env, args.model_file, 25)

    env.close()

"""
MAIN
"""
if __name__ == '__main__':
    main()