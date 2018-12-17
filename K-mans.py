from __future__ import division
import numpy as np
import scipy.io


class Cluster:
	def __init__(self, center):
		self.points = []
		self.points.append(center)
		self.center = center
		self.mean = center
		self.size = 1

	def addPoint(self, point):
		self.points.append(point)
		self.size = self.size + 1
		self.mean = ((self.mean * self.size) + point)/(self.size + 1)

	def distToCluster(self, point):
		dist = sum((np.subtract(self.center, point))**2)
		return dist

	def update(self):
		self.center = self.mean
		return self.center
	def clean(self):
		self.points = []
		



def initial(k, data):
	n = np.shape(data)[0]
	centers = [None]*k
	indices = np.random.randint(n, size = k)
	for i in range(k):
		centers[i] = data[indices[i]]
	print "centers:", indices	
	return centers

def kmeans(data, clusters):

	for point in data:
		minDist = 100000000
		min = 0
		for i in range (len(clusters)):
			if clusters[i].distToCluster(point) < minDist:
				min = i
				minDist = clusters[i].distToCluster(point)
		clusters[min].addPoint(point)

	return clusters

def evalClustering(clusters):
	scatter = 0
	for cluster in clusters:
		for point in cluster.points:
			scatter = scatter + sum((np.subtract(cluster.center, point))**2)
	return scatter



def main():
	data = scipy.io.loadmat('iris.mat')['data']
	k = int(raw_input("How many clusters do you want? "))
	centers = initial(k, data)
	clusters = []
	cost = []
	for center in centers:
		clusters.append(Cluster(center))
	for j in range(5):
		for i in range(150):
			clusters = kmeans(data, clusters)
			cost.append(evalClustering(clusters))
			if i < 149:
				for cluster in clusters:
					cluster.update()
					cluster.clean()

	for cluster in clusters:
		print "points:", len(cluster.points)
		print "mean:", cluster.mean
		print "********************************"
	mean = sum(cost)/len(cost)
	print "cost mean:", mean  
	miu = 0
	sm = 0
	for c in cost:
		sm = sm + (c-mean)**2
	miu = np.sqrt(sm)/len(cost)
	print "cost variance:", miu


if __name__ == '__main__':
	main()
