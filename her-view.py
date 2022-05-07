import gym
from stable_baselines3 import SAC, HerReplayBuffer

# Init env
env = gym.make('FetchSlide-v1')

# Init model
model = SAC.load('./her', env=env)

obs = env.reset()
for _ in range(100):
    env.render()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)

    if done:
        obs = env.reset()