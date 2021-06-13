import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

SIZEX = 4
SIZEY = 4
DISCOUNT = 1
ACTIONS = [[-1, 0], [0, -1], [1, 0], [0, 1]]
POLICY_INIT = 0.25
k = 1000


def get_reward_newState(i, j, action) :
    next_i = i + action[0]
    next_j = j + action[1]
    if ((i == 0 and j == 0) or (i == SIZEX - 1 and j == SIZEY - 1)) :
        return 0, (i, j)
    if (next_i < 0 or next_i >= SIZEX or next_j < 0 or next_j >= SIZEY) :
        return -1, (i, j)
    return -1, (next_i, next_j)

value = np.zeros((SIZEX, SIZEY))

for step in range(k) :
    new_value = np.zeros((SIZEX, SIZEY))
    for i in range(SIZEX) :
        for j in range(SIZEY) :
            for action in ACTIONS :
                reward, (new_i, new_j) = get_reward_newState(i, j, action)
                new_value[i][j] += POLICY_INIT * (reward + DISCOUNT * value[new_i][new_j])
    #check convergence
    """
    if (np.sum(np.abs(value - new_value)) < 1e-4) :
        print(new_value)
        break
    """
    value = new_value
print(value)
