#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
import gym
from gym.wrappers import FlattenObservation

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
    flatten = FlattenObservation(env)
    n_games = 1
    
    

    for _ in range(n_games):
        done = False
        obs = env.reset()
        obs_f = flatten.observation(obs)

        while not done:
            env.render()

            action = env.action_space.sample()
            obs_, reward, done, info = env.step(action)
            

        env.close()



"""
MAIN
"""
if __name__ == '__main__':
    main()