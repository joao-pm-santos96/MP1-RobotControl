import numpy as np
import gym

TWEEKS = 0.02
STAIRS = 13
BRAINS = STAIRS*(STAIRS+1)//2 

env = gym.make('FetchReach-v1')



class Normalizer():

    def __init__(self, nb_inputs):
        self.n = 0
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)
        self.fixedsqrtvar = 0

    def observe(self, x):
        if self.n==50000: 
            return
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)
        if self.n==50000:
            self.fixedsqrtvar = np.sqrt(self.var)


    def normalize(self, inputs):
        if self.n >= 50000:
            return (inputs - self.mean) / self.fixedsqrtvar
        return (inputs - self.mean) / np.sqrt(self.var)

norm = Normalizer([120, ]) 

def playEpisode(weigths):
    done = False
    accreward = 0
    obs = env.reset()
    while not done:
        diff = obs['desired_goal']-obs['achieved_goal']
        inp = np.concatenate([obs['observation'],obs['achieved_goal']*0.4, diff*0.4])
        newinp = np.array([inp[a]*inp[b] for a in range(15) for b in range(a+1,16)])
        norm.observe(newinp)
        newinp = norm.normalize(newinp)

        action = np.tanh(weigths@newinp.T)
        obs, reward, done, info = env.step(action)
        np.tanh(inp)

        accreward += reward
    return accreward

def mutateWeights(weights):
    return weights + np.random.normal(size=[BRAINS, 4, 120])*TWEEKS

def updateWeights(rewards, weights):
    rindexes = sorted([(r, idx) for idx,r in enumerate(rewards)], reverse=True)
    idx = 0
    weight_samples_new = np.zeros(weights.shape)
    print(f"reward: best {STAIRS}:", sum([i[0] for i in rindexes[:STAIRS]])/STAIRS)
    for s in range(STAIRS):
        for t in range(STAIRS-s):
            weight_samples_new[idx,:,:] = weights[rindexes[s][1],:,:].copy() 
            idx+=1
    return mutateWeights(weight_samples_new)



weight_samples = np.zeros([BRAINS, 4, 120])
weight_samples = mutateWeights(weight_samples)
while 1:
    rewards = [0]*BRAINS
    for i in range(BRAINS):
        rewards[i] = playEpisode(weight_samples[i,:,:])
    
    weight_samples = updateWeights(rewards, weight_samples)