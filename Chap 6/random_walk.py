import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns 

true_value = np.zeros(7)
true_value[1 : 6] = np.asarray(np.arange(1, 6)) / 6
LEFT = -1
RIGHT = 1
alpha_monte = [0.01, 0.02, 0.03, 0.04]
alpha_td = [0.05, 0.1, 0.15]
actions = [LEFT, RIGHT]
def random_action() :
	return np.random.choice(actions)

def monte_carlo_prediction(value, alpha) :
	current_state = 3
	trajectory = []
	trajectory.append(current_state)
	returns = 0
	while True :
		next_state = random_action() + current_state
		trajectory.append(next_state)
		if (next_state == 6) :
			returns = 1
			break
		if (next_state == 0) :
			returns = 0
			break
		current_state = next_state
	for state in trajectory[: -1] :
		value[state] += alpha * (returns - value[state])

def temporal_difference_prediction(value, alpha) :
	current_state = 3
	while True :
		next_state = random_action() + current_state
		reward = 0
		if (next_state == 6) :
			reward = 1
		value[current_state] += alpha * (reward + value[next_state] - value[current_state])
		if (next_state == 6 or next_state == 0) :
			break 
		current_state = next_state
def left_example_6_2() :
	value = np.zeros(7)
	value[1 : 6] = [0.5] * 5
	print(value[1 : 6])
	episodes = [0, 1, 10, 100000]
	plt.figure(1)
	for i in tqdm(range(episodes[-1] + 1)) :
		if (i in episodes) :
			plt.plot(value[0 : 6], label=str(i) + ' episodes')
		temporal_difference_prediction(value, 0.1)
	plt.plot(true_value[0 : 6], label='true values')
	plt.xlabel('state')
	plt.ylabel('estimated value')
	plt.legend()

def right_example_6_2() :
	runs = 100
	for alpha in alpha_monte :
		total_errors = np.zeros(101)
		for i in tqdm(range(runs)) :
			errors = []
			value = np.zeros(7)
			value[1 : 6] = [0.5] * 5
			for j in range(101) :
				errors.append(np.sqrt(np.sum(np.power(true_value - value, 2)) / 5.0))
				monte_carlo_prediction(value, alpha)
			total_errors += errors
		total_errors /= runs
		plt.plot(total_errors, linestyle='dashdot', label='MC' + ', alpha = %.02f' % (alpha))

	for alpha in alpha_td :
		total_errors = np.zeros(101)
		for i in tqdm(range(runs)) :
			errors = []
			value = np.zeros(7)
			value[1 : 6] = [0.5] * 5
			for j in range(101) :
				errors.append(np.sqrt(np.sum(np.power(true_value - value, 2)) / 5.0))
				temporal_difference_prediction(value, alpha)
			total_errors += errors
		total_errors /= runs
		plt.plot(total_errors, linestyle='solid', label='TD' + ', alpha = %.02f' % (alpha))
	plt.xlabel('episodes')
	plt.ylabel('RMS')
	plt.legend()

plt.figure(figsize=(10, 20))
plt.subplot(2, 1, 1)
left_example_6_2()

plt.subplot(2, 1, 2)
right_example_6_2()
plt.tight_layout()
plt.savefig('example_6_2.png')
plt.close()
