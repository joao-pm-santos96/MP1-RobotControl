from stable_baselines3 import HerReplayBuffer, TD3, SAC

class Agents():

    def __init__(self, env_name):
        self.env_name = env_name
        self.alg_name = None

    def gen_model(self, env, verbose=1, logs_folder=None):

        model = None
        if self.env_name == 'FetchReach-v1':

            buffer_params = {'goal_selection_strategy': 'future',
                            'n_sampled_goal': 4,
                            'online_sampling': True}

            policy_params = {'n_critics': 2,
                            'net_arch': [256, 256]}

            model = TD3(
                policy="MultiInputPolicy",
                env=env,
                learning_rate=1e-3,
                buffer_size=int(1e6),
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                train_freq=(1, 'episode'),
                policy_delay=2,
                gradient_steps=-1,
                replay_buffer_class = HerReplayBuffer,
                replay_buffer_kwargs= buffer_params,
                policy_kwargs=policy_params,
                optimize_memory_usage=False,
                create_eval_env=False,
                seed=None,
                tensorboard_log=logs_folder,
                verbose=verbose
            )

        elif self.env_name == 'FetchPush-v1':

            buffer_params = {'goal_selection_strategy': 'future',
                            'n_sampled_goal': 4,
                            'online_sampling': True}

            policy_params = {'n_critics': 2,
                            'net_arch': [256, 256, 256]}

            model = SAC(
                policy="MultiInputPolicy",
                env=env,
                learning_rate=1e-3,
                buffer_size=int(1e6),
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, 'episode'),
                use_sde=True,
                gradient_steps=-1,
                replay_buffer_class = HerReplayBuffer,
                replay_buffer_kwargs= buffer_params,
                policy_kwargs=policy_params,
                optimize_memory_usage=False,
                create_eval_env=False,
                seed=None,
                tensorboard_log=logs_folder,
                verbose=verbose
            )


        self.alg_name = model.__class__.__name__
        return model

    def get_model(self, file, env):

        model = None
        if self.env_name == 'FetchReach-v1':
            model = TD3.load(file, env)

        elif self.env_name == 'FetchPush-v1':
            model = SAC.load(file, env)

        return model
