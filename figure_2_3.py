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
        self.estimate += CFG.alpha * (reward - self.estimate)

def get_result(e, init_value):
    bandits = [bandit(np.random.normal(CFG.mean, CFG.variance), CFG.variance, init_value) for i in range(CFG.n)]
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
        bandits[choose].update(reward)
    return res


plt.figure(figsize=(10, 5))

res = np.zeros(CFG.t)
for tr in trange(CFG.n_try):
    res += get_result(0, 5)
print(res.shape)
res /= CFG.n_try
plt.plot(res, label = 'q1 = 5, e = 0')
print(f'done case 1')

res = np.zeros(CFG.t)
for tr in trange(CFG.n_try):
    res += get_result(0.1, 0)
print(res.shape)
res /= CFG.n_try
plt.plot(res, label = 'q1 = 0, e = 0.1')
print(f'done case 2')

plt.xlabel('Step')
plt.ylabel('Optimal option')
plt.legend()
plt.savefig('Figure 2_5.png')
plt.show()