import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns 
import copy

#-1 is go left
# 1 is go right
actions = []
actions.append(80702)
actions.append([-1] * 10)
actions.append([-1, 1])
alpha = 0.1
eps = 0.1
start = 2


def choose_action(state, q_value) :
	if np.random.binomial(1, eps) == 1 :
		return np.random.choice(range(len(actions[state])))
	else :
		values_ = q_value[state]
		return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

def step(state, action) :
	state += actions[state][action]
	if (state == 0) :
		return state, np.random.normal(-0.1, 1)
	return state, 0

def q_learning(q_value) :
	state = start
	action_left = 0
	while True :
		action = choose_action(state, q_value)
		if (state == start and action == 0) :
			action_left += 1 
		next_state, reward = step(state, action)
		#print(str(state) + ' ' + str(action) + ' ' + str(next_state) + ' ' + str(reward))
		q_value[state, action] += alpha * (reward + np.max(q_value[next_state]) - q_value[state, action])
		if (next_state == 0 or next_state == 3) :
			break
		state = next_state
	return action_left

def double_q_learning(q1, q2) :
	state = start
	action_left = 0
	while (True) :
		action = choose_action(state, np.asarray(q1) + np.asarray(q2))
		if (state == start and action == 0) :
			action_left += 1
		next_state, reward = step(state, action)
		if (np.random.binomial(1, 0.5) == 1) :
			q1[state, action] += alpha * (reward + q2[next_state, np.argmax(q1[next_state])] - q1[state, action])
		else :
			q2[state, action] += alpha * (reward + q1[next_state, np.argmax(q2[next_state])] - q2[state, action])
		if (next_state == 3 or next_state == 0) :
			break
		state = next_state
	return action_left
runs = 1000
episodes = 300
left_actions_q = np.zeros(episodes)
left_actions_double_q = np.zeros(episodes)
for run in tqdm(range(runs)) :
	q = np.zeros((4, len(actions[1])))
	q[start, 2 : len(actions[1])] = [-1000000] * (len(actions[1]) - 2) 
	q1 = copy.deepcopy(q)
	q2 = copy.deepcopy(q)
	left_action_q = []
	left_action_double_q = []
	for episode in range(episodes) :
		left_action_q.append(q_learning(q))
		left_action_double_q.append(double_q_learning(q1, q2))
	left_actions_q += left_action_q
	left_actions_double_q += left_action_double_q
left_actions_q /= runs
left_actions_double_q /= runs
plt.plot(left_actions_q, label = 'Q-learning')
plt.plot(left_actions_double_q, label = 'Double-Q-learning')
plt.xlabel('Episodes')
plt.ylabel('Percent left action from A')
plt.legend()
plt.savefig('figure_6_5.png')
plt.show()
plt.close()

