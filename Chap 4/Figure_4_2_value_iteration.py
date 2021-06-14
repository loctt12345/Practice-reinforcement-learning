import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson

COST_RENT = 10
COST_MOVE = 2
DISCOUNT = 0.9
LAM_REQUEST_FIRST = 3
LAM_REQUEST_SECOND = 4
LAM_RETURN_FIRST = 3
LAM_RETURN_SECOND = 2
MAX_ACTION = 5
MAX_CAR = 20
poisson_cache = dict()
actions = np.arange(-MAX_ACTION, MAX_ACTION+ 1)

def poisson_probability(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]

def solve(first_cars, second_cars, action, old_value) :
	first_cars -= action
	second_cars += action
	first_cars = min(MAX_CAR, first_cars)
	second_cars = min(MAX_CAR, second_cars)
	#cost for moving cars
	res = - abs(action) * COST_MOVE
	for request_first in range(11) :
		for request_second in range(11) :
			#probability to recevie first location request and second location request
			prob = poisson_probability(request_first, LAM_REQUEST_FIRST) * poisson_probability(request_second, LAM_REQUEST_SECOND)
			x = min(request_first, first_cars)
			y = min(request_second, second_cars)
			#number of cars after renting out
			first_cars_after = first_cars - x
			second_cars_after = second_cars - y
			#reward after renting out
			reward = (x + y) * COST_RENT
			#cars are returned
			return_first = LAM_RETURN_FIRST
			return_second = LAM_RETURN_SECOND
			#number of cars after returning
			first_cars_after = min(MAX_CAR, first_cars_after + return_first)
			second_cars_after = min(MAX_CAR, second_cars_after + return_second) 
			#update value
			res += prob * (reward + DISCOUNT * old_value[first_cars_after][second_cars_after])
	return res

#policy iteration algorithm : 

def policy_iteration() :
	print('Start')
	policy = np.zeros((MAX_CAR + 1, MAX_CAR + 1), dtype = np.int)
	value = np.zeros((MAX_CAR + 1, MAX_CAR + 1))
	while True :
		#policy evaluation
		while True :
			old_value = value.copy()
			for i in range(MAX_CAR + 1) :
				for j in range(MAX_CAR + 1) :
					for action in actions :
						if (0 <= action <= i) or (-j <= action <= 0):
					 		value[i][j] = max(value[i][j], solve(i, j, action, value))
					#print(new_value[i][j])
			max_value_change = abs(old_value - value).max()
			print('max value change {}'.format(max_value_change))
			if max_value_change < 1e-4 :
				break

		#policy improvement
		stable = True
		for i in range(MAX_CAR + 1) :
			for j in range(MAX_CAR + 1) :
				list_action = []
				for action in actions :
					 if (0 <= action <= i) or (-j <= action <= 0):
					 	list_action.append(solve(i, j, action, value))
					 else :
					 	list_action.append(-np.inf)
				new_action = actions[np.argmax(list_action)]
				if (new_action != policy[i][j]) :
					stable = False
				policy[i][j] = new_action
		print('policy stable {}'.format(stable))
		print(policy)
		#if (stable == True) :
		break
#-----------------------------------------------main-------------------------------------------------
if __name__ == '__main__':
	policy_iteration()


# Optimal policy
""" 
[[ 0  0  0  0  0  0  0  0 -1 -1 -2 -2 -2 -3 -3 -3 -3 -3 -4 -4 -4] 
[ 0  0  0  0  0  0  0  0  0 -1 -1 -1 -2 -2 -2 -2 -2 -3 -3 -3 -3] 
[ 0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -1 -1]
 [ 1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1] 
 [ 1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 
 [ 2  2  1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 
 [ 3  2  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 3  3  3  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 
 [ 4  4  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 
 [ 5  4  3  3  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 
 [ 5  4  4  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
  [ 5  5  4  3  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 
  [ 5  5  4  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 
  [ 5  5  5  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 
  [ 5  5  5  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0] 
  [ 5  5  5  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
   [ 5  5  5  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    [ 5  5  5  4  3  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0] 
    [ 5  5  5  4  3  2  2  1  1  1  1  1  1  0  0  0  0  0  0  0  0] 
[ 5  5  5  4  3  3  2  2  2  2  2  2  1  1  1  1  1  1  0  0  0]]"""