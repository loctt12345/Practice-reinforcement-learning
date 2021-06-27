import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns 

discount = 1
#####0 is stop, 1 is hit
action = [0, 1]
#####policy player
policy_player = np.zeros(22)
for i in range(12, 20) :
	policy_player[i] = 1
policy_player[20] = 0
policy_player[21] = 0
#####policy dealer 
policy_dealer = np.zeros(22)
for i in range(17) :
	policy_dealer[i] = 1
for i in range(17, 22) :
	policy_dealer[i] = 0

def get_card() :
	card = np.random.randint(1, 14)
	card = min(card, 10)
	return card

def get_value_card(card) :
	if card == 1 :
		return 11
	else :
		return card

def random_card(is_player) :
	sum_card_player = 0
	usable_ace_player = 0
	dealer_card1 = 0
	if (is_player) :
		while (sum_card_player < 12) :
			card_current = get_card()
			sum_card_player += get_value_card(card_current)
			if (sum_card_player > 21) :
				sum_card_player -= 10
			else :
				if (card_current == 1) :
					usable_ace_player = 1
	else :
		card_current = get_card()
		dealer_card1 = card_current
		sum_card_player += get_value_card(card_current)
		if (card_current == 1) :
			usable_ace_player = 1
		card_current = get_card()
		sum_card_player += get_value_card(card_current)
		if (card_current == 1) :
			usable_ace_player = 1
		if (sum_card_player > 21) :
			sum_card_player -= 10
	return sum_card_player, usable_ace_player, dealer_card1

def generate() :
	#####random player's card
	sum_card_player ,usable_ace_player, _ = random_card(True)
	#####random dealer's card
	sum_card_dealer ,usable_ace_dealer, dealer_card1 = random_card(False)
	#print(sum_card_player, usable_ace_player, sum_card_dealer, usable_ace_dealer)
	#####player's turn
	ace = usable_ace_player
	player_trajectory = []
	while (True) :
		action = policy_player[sum_card_player]
		player_trajectory.append([(sum_card_player, dealer_card1, usable_ace_player), action])
		#print('player turn :', sum_card_player, action, ace, end = ' ')
		if (action == 0) :
			break
		card_current = get_card()
		#print(card_current)
		if (card_current == 1) :
			ace += 1
		sum_card_player += get_value_card(card_current)
		if (sum_card_player > 21) :
			if (ace > 0) :
				sum_card_player -= 10
				ace -= 1
			else :
				#go bust
				return player_trajectory, -1
		usable_ace_player = 1 if (ace > 0) else 0
	#####dealer's turn
	#print('\n')
	ace = usable_ace_dealer
	while (True) :
		action = policy_dealer[sum_card_dealer]
		#print('dealer turn :', sum_card_dealer, action, ace, end = ' ')
		if (action == 0) :
			break
		card_current = get_card()
		#print(card_current)
		if (card_current == 1) :
			ace += 1
		sum_card_dealer += get_value_card(card_current)
		if (sum_card_dealer > 21) :
			if (ace > 0) :
				sum_card_dealer -= 10
				ace -= 1
			else :
				#####go bust
				return player_trajectory, 1
	if (sum_card_player > sum_card_dealer) :
		return player_trajectory, 1
	if (sum_card_player == sum_card_dealer) :
		return player_trajectory, 0
	return player_trajectory, -1

def monte_carlo_prediction(episodes) :
	states_usable_ace = np.zeros((10, 10))
	states_usable_ace_count = np.ones((10, 10))
	states_no_usable_ace = np.zeros((10, 10))
	states_no_usable_ace_count = np.ones((10, 10))
	for i in tqdm(range(0, episodes)):
		player_trajectory, reward = generate()
		for (player_sum, dealer_card, usable_ace), _ in player_trajectory:
			player_sum -= 12
			dealer_card -= 1
			if usable_ace:
				states_usable_ace_count[player_sum, dealer_card] += 1
				states_usable_ace[player_sum, dealer_card] += reward
			else:
				states_no_usable_ace_count[player_sum, dealer_card] += 1
				states_no_usable_ace[player_sum, dealer_card] += reward
	return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count


#---------------------------------------------------------------------------------
Usable10k, NoUsable10k = monte_carlo_prediction(10000)
Usable500k, NoUsable500k = monte_carlo_prediction(500000)

ax = sns.heatmap(NoUsable10k, cmap="YlGnBu", xticklabels=range(1, 11)
    ,yticklabels=list(range(12, 22)))
plt.savefig('figure_5_1_10k_NoUsable.png')
plt.close()

ax = sns.heatmap(Usable10k, cmap="YlGnBu", xticklabels=range(1, 11)
    ,yticklabels=list(range(12, 22)))
plt.savefig('figure_5_1_10k_Usable.png')
plt.close()

ax = sns.heatmap(NoUsable500k, cmap="YlGnBu", xticklabels=range(1, 11)
    ,yticklabels=list(range(12, 22)))
plt.savefig('figure_5_1_500k_NoUsable.png')
plt.close()

ax = sns.heatmap(Usable500k, cmap="YlGnBu", xticklabels=range(1, 11)
    ,yticklabels=list(range(12, 22)))
plt.savefig('figure_5_1_500k_Usable.png')
plt.close()