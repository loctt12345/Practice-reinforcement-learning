import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns 

#0 is left , 1 is right
actions = [0, 1]

def behavior_policy() :
	return np.random.choice(actions)
def target_policy() :
	return 0

def generate() :
	player_trajectory = []
	while (True) :
		action = behavior_policy()
		player_trajectory.append(action)
		if (action == 1) :
			return player_trajectory, 0
		if (np.random.binomial(1, 0.9) == 0) :
			return player_trajectory, 1

def monte_carlo_off_policy(episodes) :
	rewards = 0
	values = []
	for episode in range(episodes) :
		player_trajectory, reward = generate()
		numerator = 1.0
		denominator = 1.0
		for action in player_trajectory :
			if (action == target_policy()) :
				denominator *= 0.5
			else :
				numerator = 0.0
				break
		ratio = numerator / denominator
		rewards += ratio * reward
		values.append(rewards)
	values = np.asarray(values)
	values /= np.arange(1, episodes + 1)
	return values

for run in tqdm(range(10)) :
	values = monte_carlo_off_policy(100000)
	plt.plot(values)
plt.xlabel('Episodes (log scale)')
plt.ylabel('Ordinary Importance Sampling')
plt.xscale('log')
plt.savefig('figure_5_4.png')
plt.close()