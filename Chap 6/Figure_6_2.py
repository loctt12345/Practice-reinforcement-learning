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

def monte_carlo_prediction(value, alpha, trajectories, rewards_mc) :
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
	rewards_mc.append(returns)
	trajectories.append(trajectory)
	while True :
		updates = np.zeros(7)
		for trajectory, returns in zip(trajectories, rewards_mc) :
			for state in trajectory[: -1] :
				updates[state] += returns - value[state]
		updates *= alpha
		if (np.sum(np.abs(updates)) < 1e-3) :
			break
		value += updates

def temporal_difference_prediction(value, alpha, trajectories, rewards_td) :
	current_state = 3
	trajectory = []
	rewards = []
	while True :
		next_state = random_action() + current_state
		reward = 0
		if (next_state == 6) :
			reward = 1
		trajectory.append(current_state)
		rewards.append(reward)
		if (next_state == 6 or next_state == 0) :
			trajectory.append(next_state)
			break 
		current_state = next_state
	trajectories.append(trajectory)
	rewards_td.append(rewards)
	while True :
		updates = np.zeros(7)
		for trajectory, rewards in zip(trajectories, rewards_td) :
			for i in range(len(trajectory) - 1) :
				updates[trajectory[i]] += rewards[i] + value[trajectory[i + 1]] - value[trajectory[i]]
		updates *= alpha
		if (np.sum(np.abs(updates)) < 1e-3) :
			break
		value += updates
	
episodes =  100 + 1
runs = 100
td_erros = np.zeros(episodes)
mc_erros = np.zeros(episodes)
for run in tqdm(range(runs)) :
	value_td = np.zeros(7)
	value_td[1 : 6] = [0.5] * 5
	value_mc = np.zeros(7)
	value_mc[1 : 6] = [0.5] * 5
	errors_td = []
	errors_mc = []
	trajectories_td = []
	trajectories_mc = []
	rewards_td = []
	rewards_mc = []
	for episode in (range(episodes)) :
		temporal_difference_prediction(value_td, 0.001, trajectories_td, rewards_td)
		monte_carlo_prediction(value_mc, 0.001, trajectories_mc, rewards_mc)
		errors_td.append(np.sqrt(np.sum(np.power(value_td - true_value, 2)) / 5.0 ))
		errors_mc.append(np.sqrt(np.sum(np.power(value_mc - true_value, 2)) / 5.0 ))
	td_erros += errors_td
	mc_erros += errors_mc
	#plt.plot(value_td[0 : 6], label = 'TD')
	#plt.plot(value_mc[0 : 6], label = 'MC')

td_erros /= runs
mc_erros /= runs
plt.plot(td_erros, label='TD')
plt.plot(mc_erros, label='MC')
plt.xlabel('episodes')
plt.ylabel('RMS error')
plt.legend()
plt.savefig('figure_6_2.png')
plt.show()
plt.close()