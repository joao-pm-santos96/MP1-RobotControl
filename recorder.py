import gym
from gym.wrappers import RecordVideo
from stable_baselines3 import HerReplayBuffer, TD3, SAC, DDPG

BASE_NAME = './test_2022_05_14_20_13'

for env_name in ['FetchReach-v1', 'FetchSlide-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1']:

    for alg in [SAC, TD3, DDPG]:

        for i in range(1000,25000+1, 1000):

            file = f'{BASE_NAME}/models/{env_name}/{alg.__name__}/model_{i}_steps.zip'
            save_folder = f'{BASE_NAME}/videos/{env_name}/{alg.__name__}'

            print(file)

            env = gym.make(env_name)
            env = RecordVideo(env, video_folder=save_folder, name_prefix=f'model_{i}_steps')

            model = alg.load(file, env)
            
            obs = env.reset()
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)

            env.close()

