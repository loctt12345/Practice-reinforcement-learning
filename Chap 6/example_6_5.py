import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns 

m_col = 10
n_row = 7
winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
#(1, 0) is go down  - 0
#(0, 1) is go right - 1
#(-1, 0) is go up - 2
#(0, -1) is go left - 3
actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
start = (3, 0)
goal = (3, 7)
eps = 0.1
alpha = 0.5

def step(state, action) :
	i = state[0] + actions[action][0] - winds[state[1]]
	j = state[1] + actions[action][1]
	i = max(i, 0)
	i = min(i, n_row - 1)
	j = max(j, 0)
	j = min(j, m_col - 1)
	return (i, j)
def choose_action(state, q_value) :
	if np.random.binomial(1, eps) == 1 :
		return np.random.choice(range(4))
	else :
		values_ = q_value[state[0], state[1], :]
		return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
def play(q_value) :
	state = start
	time = 0
	action = choose_action(state, q_value)
	while True :
		next_state = step(state, action)
		reward = -1
		next_action = choose_action(next_state, q_value)
		q_value[state[0], state[1], action] += alpha * (reward + q_value[next_state[0], next_state[1], next_action] \
											- q_value[state[0], state[1], action])
		if (next_state == goal) :
			break
		time += 1
		state = next_state
		action = next_action
	return time
q_value = np.zeros((n_row, m_col, 4))
episode_limit = 170

steps = []
for ep in tqdm(range(episode_limit)) :
    steps.append(play(q_value))
print(steps)

steps = np.add.accumulate(steps)

plt.plot(steps, np.arange(1, len(steps) + 1))
plt.xlabel('Time steps')
plt.ylabel('Episodes')
plt.savefig('figure_6_3.png')
plt.show()
plt.close()
