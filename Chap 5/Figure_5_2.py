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

def get_card() :
	card = np.random.randint(1, 14)
	card = min(card, 10)
	return card

def get_value_card(card) :
	if card == 1 :
		return 11
	else :
		return card

def generate(state_init, action_init, policy) :
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
	start = True
	while (True) :
		if (start == True) :
			action = action_init
		else :
			action = policy(sum_card_player, dealer_card1, usable_ace_player)
		start = False
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

def monte_carlo_es(episodes) :
	q = np.zeros((10, 10, 2, 2))
	q_count = np.ones((10, 10, 2, 2))
	def policy_init(sum_card_player, dealer_card1, usable_ace_player) :
		return policy_player[sum_card_player]
	def policy_greedy(sum_card_player, dealer_card1, usable_ace_player) :
		sum_card_player -= 12
		dealer_card1 -= 1
		tmp = q[sum_card_player, dealer_card1, usable_ace_player, :] / q_count[sum_card_player, dealer_card1, usable_ace_player, :]
		if (tmp[0] == tmp[1]) :
			return np.random.choice(range(0, 2))
		else :
			return np.argmax(tmp)
	for episode in tqdm(range(0, episodes)) :
		state_init = [ np.random.choice(range(12, 22)), np.random.choice(range(1, 11)), np.random.choice([0, 1])]
		action_init = np.random.choice(actions)
		policy = policy_greedy if (episode) else policy_init
		player_trajectory, reward =  generate(state_init, action_init, policy)
		mark = set()
		for (sum_card_player, dealer_card1, usable_ace_player), action in player_trajectory :
			sum_card_player -= 12
			dealer_card1 -= 1
			action = int(action)
			usable_ace_player = int(usable_ace_player)
			state_action = (sum_card_player, dealer_card1, usable_ace_player, action)
			if (state_action in mark) :
				continue
			mark.add(state_action)
			q[sum_card_player, dealer_card1, usable_ace_player, action] += reward
			q_count[sum_card_player, dealer_card1, usable_ace_player, action] += 1
	return q / q_count

state_action_values = monte_carlo_es(500000)
print(state_action_values)
state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=-1)
state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis=-1)

# get the optimal policy
action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)

images = [action_usable_ace,
          state_value_usable_ace,
          action_no_usable_ace,
          state_value_no_usable_ace]

titles = ['Optimal policy with usable Ace',
          'Optimal value with usable Ace',
          'Optimal policy without usable Ace',
          'Optimal value without usable Ace']

_, axes = plt.subplots(2, 2, figsize=(40, 30))
plt.subplots_adjust(wspace=0.1, hspace=0.2)
axes = axes.flatten()

for image, title, axis in zip(images, titles, axes):
    fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                      yticklabels=list(reversed(range(12, 22))))
    fig.set_ylabel('player sum', fontsize=30)
    fig.set_xlabel('dealer showing', fontsize=30)
    fig.set_title(title, fontsize=30)

plt.savefig('figure_5_2.png')
plt.close()

"""
[[[[-0.52319109 -0.79568234]
   [-0.22496148 -0.7732042 ]]

  [[-0.20428462 -0.32112237]
   [ 0.09090909 -0.33253012]]

  [[-0.22494262 -0.22310606]
   [ 0.06807131 -0.29727498]]

  [[-0.17133132 -0.20488574]
   [ 0.14460784 -0.23047473]]

  [[-0.17970565 -0.15479018]
   [ 0.10950081 -0.159699  ]]

  [[-0.16603487 -0.19209915]
   [ 0.16251006 -0.10567211]]

  [[-0.22352941 -0.48507463]
   [ 0.16311475 -0.51661392]]

  [[-0.25954493 -0.56060606]
   [ 0.09976434 -0.55363036]]

  [[-0.32176537 -0.55968379]
   [ 0.03532182 -0.57515823]]

  [[-0.41209617 -0.57798892]
   [-0.18492063 -0.61783961]]]


 [[[-0.56361725 -0.76738411]
   [-0.29204204 -0.78247012]]

  [[-0.28247096 -0.28892006]
   [ 0.09687262 -0.29828851]]

  [[-0.3168693  -0.23917362]
   [ 0.06423358 -0.24784314]]

  [[-0.24849398 -0.2076099 ]
   [ 0.13769752 -0.20791246]]

  [[-0.27055068 -0.17483846]
   [ 0.13582966 -0.15411255]]

  [[-0.20237213 -0.16      ]
   [ 0.16161616 -0.14537108]]

  [[-0.28147612 -0.49077787]
   [ 0.09737249 -0.46966115]]

  [[-0.33545727 -0.51498423]
   [ 0.04664723 -0.52682146]]

  [[-0.39056743 -0.5188755 ]
   [-0.03981623 -0.57417803]]

  [[-0.47018432 -0.57237386]
   [-0.17030568 -0.56022187]]]


 [[[-0.58574921 -0.75419463]
   [-0.34722222 -0.7805668 ]]

  [[-0.35559131 -0.30294627]
   [ 0.01614035 -0.27908805]]

  [[-0.37598116 -0.26039305]
   [ 0.01861702 -0.24173228]]

  [[-0.32350532 -0.18858195]
   [ 0.11904762 -0.23130301]]

  [[-0.26767276 -0.18365103]
   [ 0.09214092 -0.17817896]]

  [[-0.325      -0.16666667]
   [ 0.14817416 -0.18462758]]

  [[-0.3277027  -0.47484756]
   [ 0.04605722 -0.4742268 ]]

  [[-0.37491289 -0.5       ]
   [ 0.02544704 -0.54581359]]

  [[-0.42251514 -0.52126837]
   [-0.0625     -0.51572848]]

  [[-0.51224863 -0.54146676]
   [-0.19427811 -0.60273973]]]


 [[[-0.60492185 -0.76935355]
   [-0.31916329 -0.77696078]]

  [[-0.43478261 -0.29595782]
   [-0.04457364 -0.28416667]]

  [[-0.39787092 -0.26518219]
   [ 0.00814536 -0.26875515]]

  [[-0.41185771 -0.23046453]
   [ 0.06288917 -0.20542317]]

  [[-0.36893204 -0.15275762]
   [ 0.12411576 -0.1391097 ]]

  [[-0.35851852 -0.15072035]
   [ 0.16764133 -0.17271293]]

  [[-0.39237148 -0.47885076]
   [ 0.05476804 -0.45031546]]

  [[-0.44131944 -0.51422156]
   [-0.0502611  -0.52483974]]

  [[-0.46899478 -0.53959627]
   [-0.1297043  -0.53936087]]

  [[-0.52819549 -0.58008261]
   [-0.25440806 -0.5629085 ]]]


 [[[-0.65904239 -0.77507837]
   [-0.39229025 -0.75115562]]

  [[-0.48866302 -0.30120069]
   [-0.01868596 -0.28873239]]

  [[-0.46420497 -0.2566787 ]
   [-0.05107832 -0.26036866]]

  [[-0.46142093 -0.20402194]
   [ 0.09009009 -0.20840198]]

  [[-0.45367412 -0.17937907]
   [ 0.06246385 -0.19860357]]

  [[-0.41012085 -0.139254  ]
   [ 0.09706411 -0.15672783]]

  [[-0.38840858 -0.50504202]
   [-0.00870106 -0.46048387]]

  [[-0.43326118 -0.5380334 ]
   [-0.09277745 -0.55617978]]

  [[-0.52854442 -0.55029271]
   [-0.16238245 -0.52992126]]

  [[-0.58880937 -0.58042705]
   [-0.25014715 -0.57306122]]]


 [[[-0.69287289 -0.63151515]
   [-0.42269737 -0.64544722]]

  [[-0.56916667 -0.16595289]
   [-0.01220866 -0.12861492]]

  [[-0.52520325 -0.14401773]
   [ 0.01058496 -0.15349544]]

  [[-0.51391162 -0.06945429]
   [ 0.07456873 -0.08435583]]

  [[-0.51592357 -0.04582409]
   [ 0.07312826 -0.01617647]]

  [[-0.5243811   0.01055966]
   [ 0.11540648  0.0275    ]]

  [[-0.4875502  -0.10502693]
   [ 0.05510968 -0.11019737]]

  [[-0.52193646 -0.36277705]
   [-0.07302452 -0.36683007]]

  [[-0.54894283 -0.40342146]
   [-0.13940724 -0.46425703]]

  [[-0.59525631 -0.46760294]
   [-0.26187783 -0.46953125]]]


 [[[-0.75834658 -0.37156232]
   [-0.35631605 -0.3914791 ]]

  [[-0.6093617   0.13561232]
   [ 0.05846621  0.13416537]]

  [[-0.63431677  0.14864865]
   [ 0.08982512  0.15837563]]

  [[-0.61234177  0.19502868]
   [ 0.16246954  0.21428571]]

  [[-0.61014131  0.23752969]
   [ 0.13513514  0.167     ]]

  [[-0.60468876  0.29829985]
   [ 0.19595645  0.285191  ]]

  [[-0.58025682  0.38471139]
   [ 0.12520325  0.40500511]]

  [[-0.58193445  0.09980431]
   [ 0.0601626   0.08012821]]

  [[-0.61029412 -0.15986395]
   [-0.12243826 -0.15874177]]

  [[-0.67712318 -0.21613739]
   [-0.19589854 -0.20731707]]]


 [[[-0.76773188 -0.11295034]
   [-0.26538462 -0.12819203]]

  [[-0.74707846  0.38209285]
   [ 0.07686212  0.42015968]]

  [[-0.70846154  0.38873995]
   [ 0.15945513  0.43664717]]

  [[-0.73421668  0.4298725 ]
   [ 0.13543307  0.43786408]]

  [[-0.72952087  0.45302548]
   [ 0.2331483   0.42582553]]

  [[-0.69248466  0.48653773]
   [ 0.22185703  0.49056604]]

  [[-0.70248597  0.60868177]
   [ 0.22301024  0.60573477]]

  [[-0.71328125  0.58912484]
   [ 0.10843373  0.61736973]]

  [[-0.70823341  0.28968254]
   [ 0.01097179  0.28763709]]

  [[-0.73783359 -0.00205198]
   [-0.13266998 -0.02240896]]]


 [[[-0.87660256  0.14175008]
   [-0.21770335  0.1704698 ]]

  [[-0.85294118  0.63748597]
   [ 0.19002375  0.63064741]]

  [[-0.83495146  0.64187867]
   [ 0.17923763  0.63527239]]

  [[-0.85326087  0.64443598]
   [ 0.26507818  0.67503553]]

  [[-0.85621971  0.6344855 ]
   [ 0.21457166  0.64510251]]

  [[-0.85671418  0.69487083]
   [ 0.2368      0.68079226]]

  [[-0.84120837  0.78611632]
   [ 0.25381526  0.77251407]]

  [[-0.8463357   0.79415715]
   [ 0.17942387  0.79134745]]

  [[-0.86468647  0.76373091]
   [ 0.12298558  0.75963303]]

  [[-0.85759494  0.43444026]
   [-0.02285714  0.44007237]]]


 [[[-0.99918167  0.64110525]
   [-0.08071025  0.65545494]]

  [[-0.99919679  0.88586546]
   [ 0.23906486  0.8705728 ]]

  [[-0.99919549  0.88630999]
   [ 0.25914149  0.88918919]]

  [[-0.99914749  0.88572499]
   [ 0.28582616  0.87719298]]

  [[-0.99920446  0.89232624]
   [ 0.27701056  0.89537494]]

  [[-0.99920382  0.89992361]
   [ 0.32102729  0.91181902]]

  [[-0.9991942   0.93054203]
   [ 0.31322506  0.91929666]]

  [[-0.99918963  0.92287582]
   [ 0.24220033  0.93224299]]

  [[-0.99919225  0.94876847]
   [ 0.11950655  0.94349005]]

  [[-0.99918567  0.89363167]
   [ 0.07855974  0.88055062]]]]
"""