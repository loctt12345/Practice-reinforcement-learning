import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns 

discount = 1
#####0 is stop, 1 is hit
actions = [0, 1]
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

def policy_target(sum_card_player, dealer_card1, usable_ace_player) :
  return policy_player[sum_card_player] 
def policy_behavior(sum_card_player, dealer_card1, usable_ace_player) :
  return np.random.choice(actions)

def get_card() :
	card = np.random.randint(1, 14)
	card = min(card, 10)
	return card

def get_value_card(card) :
	if card == 1 :
		return 11
	else :
		return card

def generate(state_init, policy) :
  #####init player's card
  sum_card_player ,usable_ace_player = state_init[0], state_init[2]
  #####init dealer's card
  dealer_card1 = state_init[1]
  usable_ace_dealer = 0
  if (dealer_card1 == 1) :
    usable_ace_dealer = 1
  card_current = get_card()
  sum_card_dealer = get_value_card(dealer_card1) + get_value_card(card_current)
  if (card_current == 1) :
    usable_ace_dealer = 1
  if (sum_card_dealer > 21) :
    sum_card_dealer -= 10
  #print(sum_card_player, usable_ace_player, sum_card_dealer, usable_ace_dealer)
  #####player's turn
  ace = usable_ace_player
  player_trajectory = []
  while (True) :
    action = policy(sum_card_player, dealer_card1, usable_ace_player)
    player_trajectory.append([(sum_card_player, dealer_card1, usable_ace_player), action])
    #print('player turn :', sum_card_player, action, ace, end = ' ')
    if (action == 0) :
      break
    card_current = get_card()
    #print(card_current)
    if (card_current == 1) :
      ace += 1
    sum_card_player += get_value_card(card_current)
    while (sum_card_player > 21) :
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
    while (sum_card_dealer > 21) :
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

def monte_carlo_off_policy(episodes) :
  q = np.zeros((10, 10, 2, 2))
  c = np.zeros((10, 10, 2, 2))
  state_init = [13, 2, 1]
  values_ordinary = []
  values_weighted = []
  numerator_of_value = 0
  denominator_of_value_ordinary = 0
  denominator_of_value_weighted = 0
  for episode in range(episodes) :
    player_trajectory, reward = generate(state_init, policy_behavior)
    numerator = 1.0
    denominator = 1.0
    for (sum_card_player, dealer_card1, usable_ace_player), action in player_trajectory :
      if (action == policy_target(sum_card_player, dealer_card1, usable_ace_player)) :
        denominator *= 0.5
      else :
        numerator = 0.0
        break
    ratio = numerator / denominator
    numerator_of_value += ratio * reward
    denominator_of_value_ordinary += 1
    denominator_of_value_weighted += ratio
    value_ordinary = numerator_of_value / denominator_of_value_ordinary
    if (denominator_of_value_weighted == 0) :
      value_weighted = 0
    else :
      value_weighted = numerator_of_value / denominator_of_value_weighted
    values_ordinary.append(value_ordinary)
    values_weighted.append(value_weighted)
  return values_ordinary, values_weighted

true_value = -0.27726
error_values_ordinary = np.zeros(10000)
error_values_weighted = np.zeros(10000)
for run in tqdm(range(100)) :
  values_ordinary, values_weighted = monte_carlo_off_policy(10000)
  x = np.asarray(values_weighted) - true_value 
  y = np.asarray(values_ordinary) - true_value
  error_values_ordinary += np.power(y, 2) 
  error_values_weighted += np.power(x, 2) 
error_values_ordinary /= 100
error_values_weighted /= 100
plt.plot(np.arange(1, 10001), error_values_ordinary, color='green', label='Ordinary Importance Sampling')
plt.plot(np.arange(1, 10001), error_values_weighted, color='red', label='Weighted Importance Sampling')
plt.ylim(-0.1, 5)
plt.xlabel('Episodes (log scale)')
plt.ylabel(f'Mean square error\n(average over 100 runs)')
plt.xscale('log')
plt.legend()
plt.savefig('figure_5_3.png')
plt.close()
