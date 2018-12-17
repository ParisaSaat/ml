import pandas as pd
import matplotlib.pyplot as plt
from numpy import *



def cost_function(teta, x_points, y_points, l):
	regularization_term = l * ((teta[1]**2) + (teta[2]**2) + (teta[3]**2) + (teta[4]**2))
	totalError = 0
	for i in range(0, len(x_points)):
		x = x_points[i]
		y = y_points[i]
		totalError += (y - (teta[4] * (x**4) + teta[3] * (x**3) + teta[2] * (x**2) + teta[1] * x + teta[0])) ** 2
	return ((totalError / len(x_points)) + regularization_term)

def step_gradient(teta_current, x_points, y_points, learning_rate, l):
	teta_gradient = [0]*5
	N = float(len(x_points))
	for i in range(0, len(x_points)):
		x = x_points[i]
		y = y_points[i]

		teta_gradient[0] -= (y - ((teta_current[4] * (x**4)) + (teta_current[3] * (x**3)) + (teta_current[2] * (x**2)) + \
			(teta_current[1] * x) + teta_current[0]))
		for i in range(1, 5):
			teta_gradient[i] -= (y - ((teta_current[4] * (x**4)) + (teta_current[3] * (x**3)) + (teta_current[2] * (x**2)) + \
				(teta_current[1] * x) + teta_current[0])) * (x**i)

	new_teta = [0]*5
	for i in range(0, 5):
		new_teta[i] = teta_current[i] - ((learning_rate/N) * teta_gradient[i])
	return new_teta

def gradient_descent_runner(x_points, y_points, starting_teta,learning_rate, num_iterations, l):
	teta = [0]*5
	for i in range(0, 5):
		teta[i] = starting_teta[i]
	for i in range(num_iterations):
		teta = step_gradient(teta, array(x_points), array(y_points), learning_rate, l)
	return teta 

def run():
	x = dataset['sepal-length']
	x_points = (x-min(x))/(max(x)-min(x))
	y = dataset['petal-length']
	y_points = (y-min(y))/(max(y)-min(y))
	learning_rate = 0.01
	initial_teta = [0]*5
	num_iterations = 1500
	l = 0.5
	teta = gradient_descent_runner(x_points, y_points, initial_teta, learning_rate, num_iterations, l)
	print "After {0} iterations teta0 = {1}, teta1 = {2}, teta2 = {3}, teta3 = {4}, error = {5}".format(num_iterations,\
		teta[0], teta[1], teta[2], teta[3], cost_function(teta, x_points, y_points, l))
	return teta


if __name__ == "__main__":
	# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	dataset = pd.read_csv(url, names=names)
	color = []
	for index, row in dataset.iterrows():
		if row['class'] == "Iris-setosa":
			color.append("r")
		elif row['class'] == "Iris-versicolor":
			color.append("g")
		elif row['class'] == "Iris-virginica":
			color.append("b")

	# SCATTER PLOT OF SEPAL_LENGTH AND SEPAL_WIDTH
	# plt.scatter( dataset['sepal-length'],  dataset['sepal-width'], c= color)
	# plt.xlabel('sepal-length')
	# plt.ylabel('sepal-width')
	# plt.show()

	teta = run()
	x = dataset['sepal-length']
	normalized_x = (x-min(x))/(max(x)-min(x))
	width = dataset['petal-length']
	normalized_width = (width-min(width))/(max(width)-min(width))
	plt.scatter( normalized_x, normalized_width, c= color)

	test_data = arange(0,1,0.005)
	y = teta[4] * (test_data**4) + teta[3] * (test_data**3) + teta[2] * (test_data**2) + teta[1] * test_data + teta[0]
	plt.plot(test_data, y)
	plt.xlabel('sepal-length')
	plt.ylabel('petal-length')

	plt.show()
