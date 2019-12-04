from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as pl


def init_centroids(X, K):
    randidx = np.random.permutation(len(X))
    centroids = X[randidx[0:K]]
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


def kmeans(X, min_centroid, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        centroids[k] = np.mean(X[min_centroid.ravel() == k], axis=0)
    return centroids


data = loadmat('bird_small.mat')
print(data)

data = data['A']
data = data / 255

X = data.reshape(data.shape[0] * data.shape[1], 3)
k = 16
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

min_centroids = minimize_centroid(X, centroids)
X_recovered = centroids[min_centroids]
X_recovered = X_recovered.reshape(data.shape[0], data.shape[1], 3)

pl.subplot(1, 2, 1)
pl.imshow(data)
pl.subplot(1, 2, 2)
pl.imshow(X_recovered)
pl.show()

pl.show()
