import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import math 


class CFG:
    n = 10
    mean = 4.0
    variance = 1.0
    t = 1000
    n_try = 2000
    alphas = [0.1, 0.4]

class bandit():
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
        self.estimate = 0
    def get_reward(self):
        reward = np.random.randn() + self.mean
        return reward

def update_gradient(reward, action, bandits, baseline, alpha, avg, prob):
    #Gradient update
    bandits[action].estimate += alpha * (reward - avg * baseline) * (1 - prob)
    for i in range(CFG.n) :
        if i != action :
            bandits[i].estimate -= alpha * (reward - avg * baseline) * prob

def get_result_gradient(alpha, baseline):
    bandits = [bandit(np.random.randn() + CFG.mean, CFG.variance) for i in range(CFG.n)]
    res = []
    avg = 0
    best_choice = np.argmax([bandit.mean for bandit in bandits])
    for step in range(CFG.t):
        exp_est = [np.exp(bandit.estimate) for bandit in bandits]
        probs = [np.exp(bandit.estimate) / np.sum(exp_est) for bandit in bandits]
        choose = np.random.choice(np.arange(CFG.n), p = probs)
        reward = bandits[choose].get_reward()
        avg += (reward - avg) / (step + 1)
        if (choose == best_choice) :
            res.append(1)
        else :
            res.append(0)
        update_gradient(reward, choose, bandits, baseline, alpha, avg, probs[choose])
    return res

plt.figure(figsize=(10, 5))

for alpha in CFG.alphas :
    for baseline in range(2) :
        res = np.zeros(CFG.t)
        for tr in trange(CFG.n_try):
            res += get_result_gradient(alpha, baseline)
        print(res.shape)
        res /= CFG.n_try
        plt.plot(res, label = str(alpha) + ' ' + str(baseline))
        print(f'done case' )

plt.xlabel('Step')
plt.ylabel('Optimal action')
plt.legend()
plt.savefig('Figure 2_5.png')
plt.show()