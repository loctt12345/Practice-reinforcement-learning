import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


class CFG:
    n = 2
    mean = 0.0
    variance = 1.0
    t = 1000
    n_try = 200
    alpha = 0.1
    es = [0.0, 0.1, 0.01]

class bandit():
    def __init__(self, mean, variance, estimate, cnt):
        self.mean = mean
        self.variance = variance
        self.estimate = estimate
        self.cnt = cnt
    def get_reward(self):
        reward = np.random.normal(self.mean, self.variance)
        return reward

    def update(self, reward):
        self.cnt += 1
        #e-greedy with const alpha
        self.estimate += CFG.alpha * (reward - self.estimate)

def get_result(c):
    bandits = []
    res = []
    bandits.append(bandit(10, 1, 0, 0))
    bandits.append(bandit(20, 1, 0, 0))
    for step in range(CFG.t):
        if (np.random.randint(1000) % 2 == 0) :
            bandits[0] = bandit(10, 1, bandits[0].estimate, bandits[0].cnt)
            bandits[1] = bandit(20, 1, bandits[1].estimate, bandits[1].cnt)
        else :
            bandits[0] = bandit(90, 1, bandits[0].estimate, bandits[0].cnt)
            bandits[1] = bandit(80, 1, bandits[1].estimate, bandits[1].cnt)
        choose = np.argmax([bandit.estimate + c * np.sqrt(np.log(step + 1)) / (bandit.cnt + 1e-5) for bandit in bandits])
        best_choice = np.argmax([bandit.mean for bandit in bandits])
        if (choose == best_choice) :
            res.append(1)
        else :
            res.append(0)
        reward = bandits[choose].get_reward()
        bandits[choose].update(reward)
    return res


plt.figure(figsize=(10, 5))
for _ in trange(CFG.n_try) :
    res = np.zeros(CFG.t)
    res += get_result(2)
res /= CFG.n_try
plt.plot(res, label = 'f')
print(f'done')

plt.xlabel('Step')
plt.ylabel('Optimal option')
plt.legend()
plt.savefig('Exercise 2_10.png')
plt.show()