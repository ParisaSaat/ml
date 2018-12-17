from __future__ import division
import scipy.io
import numpy as np
import math
import operator
import random

def kFold(data, labels, k, i):

	dataSize = np.shape(data)[0]
	foldSize = int(dataSize / k)
	print foldSize, "==", dataSize
	index = i*foldSize
	folds = [None]*k
	trainData = []
	trainLabels = []

	testData = data[index: index + foldSize]
	testLabels = labels[index: index + foldSize]
	print index, "**", index+foldSize 
	for j in range(0, dataSize):
		if j in range(index, index + foldSize):
			continue
		else:
			trainData.append(data[j])
			trainLabels.append(labels[j])

	return trainData, trainLabels, testData, testLabels

def euclideanDis(x, y):
	n = len(x)
	sum = 0
	for i in range(n):
		sum += (x[i]-y[i])**2
	return math.sqrt(sum)

def cosineDist(x, y):
	sumx = 0
	sumy = 0
	mult = 0
	for i in range(len(x)):
		sumx = sumx + x[i]**2
		sumy = sumy + y[i]**2
		mult = mult + (x[i]*y[i])
	dist = mult/(math.sqrt(sumx) * math.sqrt(sumy))
	return 1 - dist


def majority(a):
	unique, counts = np.unique(a, return_counts=True)
	stats = dict(zip(unique, counts))
	return max(stats.iteritems(), key=operator.itemgetter(1))[0]




def KNN(k, trainData, trainLabels, testData, testLabels):
	m = np.shape(trainData)[0]
	n = np.shape(testData)[0]
	neighbors = np.zeros((n, k))
	nnLabels = np.zeros((n, k))
	distances = np.zeros((n, m))
	prediction = np.zeros(n)
	for i in range(n):
		for j in range(m):
			distances[i][j] = euclideanDis(testData[i], trainData[j])
			distances[i][j] = cosineDist(testData[i], trainData[j])



	for i in range(n):
		neighbors[i] = distances[i].argsort()[:k]
		for j in range(k):
			nnLabels[i][j] = trainLabels[int(neighbors[i][j])]
		prediction[i] = majority(nnLabels[i])

	correct = 0
	for i in range(n):
		if prediction[i] == testLabels[i]:
			correct = correct + 1
	accuracy = (correct*100)/n
	print "accuracy = ", accuracy,"%"
	return accuracy

def shuffle(data, labels):
	indices = random.sample(range(0, len(data)), len(data))
	newData = []
	newLabels = []
	for index in indices:
		newData.append(data[index])
		newLabels.append(labels[index][0])

	return newData, newLabels


def main():
	kf = int(raw_input("k for k-fold cross validation: "))
	knn = int(raw_input("How many neighbors do you want to assessment? "))
	data = scipy.io.loadmat('iris.mat')['data']
	labels = scipy.io.loadmat('iris.mat')['labels']
	data, labels = shuffle(data, labels)
	print np.shape(labels)
	print np.shape(data)
	sumAcc = [0]*5
	for j in range(0, 5):
		for i in range(kf):
			print i
			trainData, trainLabels, testData, testLabels = kFold(data, labels, kf, i)
			sumAcc[j] = sumAcc[j] + KNN(knn, trainData, trainLabels, testData, testLabels)
		sumAcc[j] = sumAcc[j]/kf
	print "final accuracy:", sum(sumAcc)/5

if __name__ == '__main__':
    main()
