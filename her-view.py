import gym
from stable_baselines3 import SAC, HerReplayBuffer

# Init env
env = gym.make('FetchSlide-v1')

# Init model
model = SAC.load('./her', env=env)

obs = env.reset()
done = False

while not done:
    env.render()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)

env.close()
