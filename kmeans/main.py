from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as pl


def init_centroids(X, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        centroids[i] = X[np.random.randint(len(X))]
    return centroids


def minimize_centroid(X, centroids):
    min_centroid = [0.0 for i in range(len(X))]
    for i in range(len(X)):
        min_distance = float('inf')
        for j in range(len(centroids)):
            distance = np.sqrt(np.sum(np.square(X[i] - centroids[j])))
            if min_distance > distance:
                min_distance = distance
                min_centroid[i] = j
    return np.array(min_centroid)


def kmeans(X, min_centroid, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        summ = np.zeros((1, X.shape[1]))
        counter = 0
        for j in range(len(X)):
            if min_centroid[j] == i:
                summ += X[j]
                counter += 1
        centroids[i] = summ/counter
    return centroids


data = loadmat('ex7data2.mat')

X = data['X']

k = 3
iterations = 10

centroids = init_centroids(X, k)
previous_centroids = centroids

for i in range(iterations):
    min_centroid = minimize_centroid(X, centroids)

    colors = [pl.cm.tab20(float(i) / 10) for i in min_centroid]
    pl.scatter(X[:, 0], X[:, 1], c=colors, s=2)
    pl.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker='x',
        c='k',
        s=100,
        linewidth=1)

    for j in range(k):
        pl.plot([
            centroids[j][0],
            previous_centroids[j][0]],
                [centroids[j][1],
                 previous_centroids[j][1]], c='k')

    previous_centroids = centroids
    centroids = kmeans(X, min_centroid, k)
    if ((centroids == previous_centroids).all()):
        break

pl.show()
