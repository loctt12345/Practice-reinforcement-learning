import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns 

#(1, 0) is go down  - 0
#(0, 1) is go right - 1
#(-1, 0) is go up - 2
#(0, -1) is go left - 3
actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
start = (3, 0)
goal = (3, 11)
n_row = 4
m_col = 12
alpha = 0.5
eps = 0.1
cliff = []
for i in range(1, m_col - 1) :
	cliff.append((3, i))

def step(state, action) :
	i = state[0] + actions[action][0]
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

def sara(q_value) :
	state = start
	rewards = 0
	action = choose_action(state, q_value)
	while True :
		next_state = step(state, action)
		reward = -1
		if (next_state in cliff) :
			next_state = start
			reward = -100
		next_action = choose_action(next_state, q_value)
		q_value[state[0], state[1], action] += alpha * (reward + q_value[next_state[0], next_state[1], next_action] \
												- q_value[state[0], state[1], action])
		rewards += reward
		if (next_state == goal) :
			break
		state = next_state
		action = next_action
	return rewards

def q_learning(q_value) :
	state = start
	rewards = 0
	while True :
		action = choose_action(state, q_value)
		next_state = step(state, action)
		#print(str(action) + ' ' + str(next_state))
		reward = -1
		if (next_state in cliff) :
			next_state = start
			reward = -100
		q_value[state[0], state[1], action] += alpha * (reward + \
			np.max(q_value[next_state[0], next_state[1], :]) - q_value[state[0], state[1], action])
		rewards += reward
		if (next_state == goal) :
			break
		state = next_state
	return rewards

episodes = 501
runs = 50
sum_rewards_sara = np.zeros(episodes)
sum_rewards_q_learning = np.zeros(episodes)
for run in tqdm(range(runs)) :
	q_value_sara = np.zeros((n_row, m_col, 4))
	q_value_q_learning = np.zeros((n_row, m_col, 4))
	sum_reward_sara= []
	sum_reward_q_learning = []
	for episode in (range(episodes)) :
		sum_reward_sara.append(sara(q_value_sara))
		sum_reward_q_learning.append(q_learning(q_value_q_learning))
	sum_rewards_sara += sum_reward_sara
	sum_rewards_q_learning += sum_reward_q_learning
sum_rewards_sara /= runs
sum_rewards_q_learning /= runs
plt.plot(sum_rewards_sara, label='Sara')
plt.plot(sum_rewards_q_learning, label='Q-learning')
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards during episode')
plt.ylim([-100, 0])
plt.legend()
plt.savefig('Example_6_6.png')
plt.show()
plt.close()