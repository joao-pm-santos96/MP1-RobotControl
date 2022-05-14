import os
import gym
from stable_baselines3 import HerReplayBuffer, TD3, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

TIMESTEPS=int(10e3)

for env_name in ['FetchReach-v1', 'FetchSlide-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1']:

    for alg in [SAC, TD3]:

        env = gym.make(env_name)

        models_folder = f'./test/models/{env_name}/{alg.__name__}'
        logs_folder = f'./test/logs/{env_name}/{alg.__name__}'

        if not os.path.exists(models_folder):
            os.makedirs(models_folder)

        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)

        buffer_params = {'goal_selection_strategy': 'future',
                        'n_sampled_goal': 4,
                        'online_sampling': True}

        policy_params = {'n_critics': 2,
                        'net_arch': [256, 256]}

        model = alg(
                policy="MultiInputPolicy",
                env=env,
                learning_rate=1e-3,
                buffer_size=int(1e6),
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, 'episode'),                gradient_steps=-1,
                replay_buffer_class = HerReplayBuffer,
                replay_buffer_kwargs= buffer_params,
                policy_kwargs=policy_params,
                optimize_memory_usage=False,
                create_eval_env=False,
                seed=None,
                tensorboard_log=logs_folder,
                verbose=0
            )

        # Callbacks
        # Save a checkpoint every x steps
        checkpoint_callback = CheckpointCallback(save_freq=500, save_path=models_folder, name_prefix='model')

        # Train the model
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=checkpoint_callback)

        # Save the best model
        model.save(f'{models_folder}/final')

        env.close()


