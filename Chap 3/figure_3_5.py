import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

SIZEX = 5
SIZEY = 5
DISCOUNT = 0.9
ACTIONS = [[-1, 0], [0, -1], [1, 0], [0, 1]]
ACTION_PROB = 0.25


def get_reward(prei, prej, i, j) :
    if (i < 0 or i >= SIZEX or j < 0 or j >= SIZEY) :
        return -1.0
    if (prei == 0 and prej == 1 and i == 4 and j == 1) :
        return 10
    if (prei == 0 and prej == 3 and i == 2 and j == 3) :
        return 5
    return 0

value = np.zeros((SIZEX, SIZEY))

while True :
    new_value = np.zeros((SIZEX, SIZEY))
    for i in range(SIZEX) :
        for j in range(SIZEY) :
            tmp = []
            for action in ACTIONS :
                if (i == 0 and j == 1) :
                    new_i, new_j = 4, 1
                elif (i == 0 and j == 3) :
                    new_i, new_j = 2, 3
                else :
                    new_i = i + action[0]
                    new_j = j + action[1]
                reward = get_reward(i, j, new_i, new_j)
                if (new_i < 0 or new_i >= SIZEX or new_j < 0 or new_j >= SIZEY) :
                    new_i, new_j = i, j
                #print(new_i, new_j)
                tmp.append(reward + DISCOUNT * value[new_i][new_j])
            new_value[i][j] = np.max(tmp)
    #check convergence
    if (np.sum(np.abs(value - new_value)) < 1e-4) :
        print(new_value)
        break
    value = new_value

