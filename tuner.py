import random
import gym
import numpy as np
import yaml

from pprint import pprint
from multiprocessing import Process, Pool
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import HerReplayBuffer, DDPG

learning_rates = [1e-4, 1e-3, 1e-2]
taus = [0.01, 0.05, 0.95, 0.97, 0.99]
gammas = [0.9, 0.95, 0.99]
action_noises = [0.0, 0.1, 0.2, 0.3]
n_layers = [2, 3]
layer_sizes = [64, 128, 256]

N = 10
N_TIMESTEPS = int(5e3)
ALG = DDPG
ENV = 'FetchReach-v1'

alg_name = str(ALG.__name__)

def eval(idx):

    print(f'Started worker {idx}')
    env = gym.make(ENV)

    # Random choices
    hyperparams = {
        'learning_rate': random.choice(learning_rates),
        'buffer_size': int(1e6),
        'learning_starts': 1000,
        'batch_size': 256,
        'tau': random.choice(taus),
        'gamma': random.choice(gammas),
        'noise': random.choice(action_noises)}

    policy_params = { 
        'net_arch': [random.choice(layer_sizes)] * random.choice(n_layers),
        'n_critics': 1}

    buffer_class = HerReplayBuffer
    buffer_params = {
        'n_sampled_goal': 4,
        'goal_selection_strategy': "future",
        'online_sampling': True}
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=hyperparams['noise'] * np.ones(n_actions))

    chosen_params = hyperparams
    chosen_params['policy_params'] = policy_params
    chosen_params['buffer_params'] = buffer_params
    pprint(chosen_params)

    # Init model
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
        action_noise=action_noise,
        replay_buffer_class = buffer_class,
        replay_buffer_kwargs= buffer_params,
        policy_kwargs=policy_params,
        optimize_memory_usage=False,
        create_eval_env=False,
        seed=None,
        tensorboard_log=f'./tuner_logs/{ENV}/{alg_name}',
        verbose=0
    )

    # Learn
    model.learn(total_timesteps=N_TIMESTEPS)

    # Test
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        score += reward

    env.close()
    print(f'Worker {idx} score: {score}')

    return (score, chosen_params)

if __name__ == '__main__':
    
    with Pool(processes=N) as pool:
        res = pool.map(eval, range(N))
        
        idx = np.argmax([i[0] for i in res])
        hyperparams = res[idx][1]
        print(hyperparams)

    with open(f"{ENV}.yaml", mode="w") as file:
        yaml.dump(hyperparams, file)





