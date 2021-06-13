import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import math 


class CFG:
    n = 10
    mean = 0.0
    variance = 1.0
    t = 1000
    n_try = 2000
    alpha = 0.1
    es = [0.0, 0.1, 0.01]
    c = 2

class bandit():
    def __init__(self, mean, variance, estimate):
        self.mean = mean
        self.variance = variance
        self.estimate = estimate
        self.cnt = 0
    def get_reward(self):
        reward = np.random.normal(self.mean, self.variance)
        return reward

    def update(self, reward):
        self.cnt += 1
        #Constant step-size parameter
        self.estimate += (1 / self.cnt) * (reward - self.estimate)

def get_result_ucb(c):
    bandits = [bandit(np.random.normal(CFG.mean, CFG.variance), CFG.variance, 0) for i in range(CFG.n)]
    res = []
    best_choice = np.argmax([bandit.mean for bandit in bandits])
    for step in range(CFG.t):
        choose = np.argmax([bandit.estimate + c * math.sqrt(math.log(step + 1) / (bandit.cnt + 1e-5)) for bandit in bandits])
        reward = bandits[choose].get_reward()
        res.append(reward)
        bandits[choose].update(reward)
    return res

def get_result_egreedy(e):
    bandits = [bandit(np.random.normal(CFG.mean, CFG.variance), CFG.variance, 0) for i in range(CFG.n)]
    res = []
    best_choice = np.argmax([bandit.mean for bandit in bandits])
    for step in range(CFG.t):
        if (np.random.random() < e):
            choose = np.random.choice(CFG.n)
        else:
            choose = np.argmax([bandit.estimate for bandit in bandits])

        reward = bandits[choose].get_reward()
        res.append(reward)
        bandits[choose].update(reward)
    return res


plt.figure(figsize=(10, 5))

res = np.zeros(CFG.t)
for tr in trange(CFG.n_try):
    res += get_result_egreedy(0.1)
print(res.shape)
res /= CFG.n_try
plt.plot(res, label = 'e-greedy e = 0.1')
print(f'done case 1')

res = np.zeros(CFG.t)
for tr in trange(CFG.n_try):
    res += get_result_ucb(2)
print(res.shape)
res /= CFG.n_try
plt.plot(res, label = 'ucb c = 2')
print(f'done case 2')

plt.xlabel('Step')
plt.ylabel('Average reward')
plt.legend()
plt.savefig('Figure 2_4.png')
plt.show()