import gym


env = gym.make('FetchSlide-v1')

# env = gym.wrappers.Monitor(env, './video/', force = True)
t = 0
while True:
   t += 1
   env.render()
   observation = env.reset()
   print(observation)
   action = env.action_space.sample()
   observation, reward, done, info = env.step(action)
   if done:
    print("Episode finished after {} timesteps".format(t+1))
    break

env.close()


