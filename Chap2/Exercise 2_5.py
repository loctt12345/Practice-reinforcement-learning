import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


class CFG:
    n = 10
    mean = 0.0
    variance = 1.0
    t = 1000
    n_try = 2000
    alpha = 0.1
    es = [0.0, 0.1, 0.01]

class bandit():
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
        self.estimate = 0.0
        self.cnt = 0
    def get_reward(self):
        reward = np.random.normal(self.mean, self.variance)
        return reward

    def update(self, reward, const_alpha):
        self.cnt += 1
        if (const_alpha == False) :
        #Avg sample
            self.estimate += (1 / self.cnt) * (reward - self.estimate)
        else :
        #Const alpha
            self.estimate += CFG.alpha * (reward - self.estimate)

def get_result(e, const_alpha):
    bandits = [bandit(np.random.normal(0, 1), CFG.variance) for i in range(CFG.n)]
    res = []
    best_choice = np.argmax([bandit.mean for bandit in bandits])
    for _ in range(CFG.t):
        if (np.random.random() < e):
            choose = np.random.choice(CFG.n)
        else:
            choose = np.argmax([bandit.estimate for bandit in bandits])

        reward = bandits[choose].get_reward()
        if choose == best_choice :
            res.append(1)
        else :
            res.append(0)
        bandits[choose].update(reward , const_alpha)
    return res


plt.figure(figsize=(10, 5))

for case in range(2) : 
    res = np.zeros(CFG.t)
    for tr in trange(CFG.n_try):
        res += get_result(0.1, bool(case))
    print(res.shape)
    res /= CFG.n_try
    plt.plot(res, label = case)
    print(f'done case {case}')

plt.xlabel('Step')
plt.ylabel('Optimal option')
plt.legend()
plt.savefig('Exercise 2_5.png')
plt.show()