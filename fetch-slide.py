#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
import gym

"""
TODO
"""

"""
CLASS DEFINITIONS
"""

"""
FUNCTIONS DEFINITIONS
"""
def main():
    env = gym.make('FetchSlide-v1')
    
    for i_episode in range(20):
        observation = env.reset()

        for t in range(100):
            env.render()
            
            action = env.action_space.sample()
            print(env.action_space)
            print(action)

            # reward scale is environment dependent, but must always increase
            observation, reward, done, info = env.step(action)
            
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    env.close()

"""
MAIN
"""
if __name__ == '__main__':
    main()