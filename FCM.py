import numpy as np
import matplotlib.pyplot as plt
import math
import random

m = 2

class Cluster:
	def __init__(self, center):
		self.points = []
		self.points.append(center)
		self.center = center
		self.mean = center
		self.size = 1
		self.newSize = []

	def addPoint(self, point):
		self.points.append(point)
		self.size = self.size + 1
		# self.mean = ((self.mean * self.size) + point)/(self.size + 1)

	def distToCluster(self, point):
		dist = sum((np.subtract(self.center, point))**2)
		return dist

	def update(self, u, data):
		sum1 = 0
		sum2 = 0
		for j in range(len(data)):
			sum1 += (u[j]**m) * data[j]
			sum2 += u[j]**m
		self.center = sum1/sum2
		print self.center
		return self.center

	def clean(self):
		self.points = []



def initial(k, data):
	n = np.shape(data)[0]
	print n
	centers = [None]*k
	max_x = max(data[0])
	min_x = min(data[0])
	max_y = max(data[1])
	min_y = min(data[1])
	 
	# for i in range(k):
	# 	centers[i] = [np.random.randint(min_x, max_x), np.random.randint(min_y, max_y)]
	centers = [[44.10053964, 65.27690199], [65.1770764, 101.02849845], [79.1523713, 65.54104598]]
	print "centers:", centers	
	clusters = []
	for center in centers:
		clusters.append(Cluster(center))

	return clusters

def FCM(k, data, clusters):
	u = np.zeros((len(clusters), len(data)))
	# u = [None]*3000
	# for i in range(len(data)):
	# 	u[:,i] = np.random.rand(k)
	# 	u[:,i] /= sum(u[:,i])
	# print np.shape(u), "------------------------"
	# for i in range(len(clusters)):
	# 		clusters[i].update(u[i], data)

	d = np.zeros(len(clusters))
	costs = []
	finish = False
	itr = 0
	while(not(finish)):
		itr += 1
		for j1 in range(len(data)):
			for i1 in range(len(clusters)):
				temp = 0
				for k in range(len(clusters)):
					# d[k] = clusters[k].distToCluster(data[j1])
					temp += 1/clusters[k].distToCluster(data[j1])
				if(not(np.array_equal(data[j1], clusters[i1].center))):
					u[i1][j1] = 1/(temp * clusters[i1].distToCluster(data[j1]))
		cost = 0
		for i2 in range(len(clusters)):
			for j2 in range(len(data)):
				cost += (u[i2][j2]**m)*clusters[i2].distToCluster(data[j2])
		costs.append(cost)
		if itr > 20 and (costs[len(costs)-1] - costs[len(costs)-2]) <= 0.0001:
			finish = True
		# if itr > 100:
		# 	finish = True

		for i3 in range(len(clusters)):
			clusters[i3].update(u[i3], data)
	print itr,"######################################"
	plt.plot(range(itr), costs)
	plt.ylabel('cost')
	plt.xlabel('iteration')
	plt.show()
	colors = []	
	for i in range(3000):
		colors.append(u[:, i])
	plt.scatter(data[:,0], data[:,1], c=colors)
	for cluster in clusters:
		plt.scatter(cluster.center[0], cluster.center[1], c=9119)
	plt.ylabel('u')
	plt.xlabel('iteration')
	plt.show()

	return u


def main():
	data = np.loadtxt('EMG_data.csv', dtype=np.float32, delimiter=',')
	data = data[:,1:]
	k = 3
	clusters = initial(k, data)
	u = FCM(k, data, clusters)
	flag = 1
	finish = 0
	newSize = {}

	for i in range(3000):
		count = 0
		_max = max(u[:,i])
		_maxIndex = np.argmax(u[:,i])
		for j in range(k):
			if _max >= u[j,i]*2:
				count += 1
		if count >= k-1:
			clusters[_maxIndex].addPoint(u[:,i])
		else:
			clusters[_maxIndex].newSize.append(u[:,i])

	for i in range(3):
		print "cluster:", i, "points=",clusters[i].size
		print "newSize:", len(clusters[i].newSize)




if __name__ == "__main__":
	main()
