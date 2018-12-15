import numpy as np
import matplotlib.pyplot as plt


theta = 0.000001
gamma = 1
states = range(1, 100) 
delta = 0
pBenefit = 0.4
v = np.zeros(201)
policy = np.zeros(101)


def reward(x):
	return (x >= 100)

def BellmanEquation(stake):
	actions =  range(0, min(stake, 100-stake)+1)
	maxValue = 0
	for a in actions:

		ifBenefit = stake + 2 * a
		ifLoss = stake - 2 * a
		sigma = pBenefit * (reward(ifBenefit) + gamma * v[ifBenefit]) + (1 - pBenefit) * (reward(ifLoss) + gamma * v[ifLoss])
		if sigma > maxValue:
			maxValue = sigma
			v[stake] = sigma
			policy[stake] = a

def valueIteration():
	delta = 1
	while delta > theta:
		delta = 0
		for s in states:
			vPrime =  v[s]
			BellmanEquation(s)
			delta = max(delta, np.abs(vPrime-v[s]))

	fig, axs = plt.subplots(2, 1)
	axs[0].plot(policy)
	axs[0].set_title('Policy')
	axs[1].plot(v[:100])
	axs[1].set_title('State Value')
	fig.suptitle('Discount Factor = 0.9', fontsize=16)

	plt.show()



def main():
	valueIteration()

if __name__ == "__main__":
    main()
