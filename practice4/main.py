import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

def mapFeature(X1, X2):
    degree = 6
    out = np.ones(X.shape[0])[:,np.newaxis]
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))[:,np.newaxis]))
    return out

def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]

def hypothesis(X, thetas):
    return 1 / (1 + np.exp(-X.dot(thetas)))

def gradient(thetas, X, Y):
    return hypothesis(X, thetas) - Y


dataRaw = np.loadtxt(fname="ex2data2.txt", delimiter=",", usecols=[0, 1, 2])
Y = [dataRaw[i][2] for i in range(len(dataRaw))]
Y = np.asarray(Y, dtype=np.float)
X = np.delete(dataRaw, 2, 1)

thetas = np.zeros(28, dtype=np.float32)

# X = X.transpose()
# pl.scatter(X[0], X[1])
# X = X.transpose()

# ax = pl.figure().gca(projection="3d")
# X = X.transpose()

X = np.array([np.insert(i, 0, 1) for i in X])
X = mapFeature(X[:,1], X[:,2])

alpha = 0.1
epochs = 100000

for i in range(epochs):
    next_thetas = np.zeros(28).astype(float)
    for j in range(28):
        if (j == 0):
            next_thetas[j] = thetas[j] - (alpha / len(X)) * np.sum((hypothesis(X, thetas) - Y) * np.array(X[:, j]))
        else:
            next_thetas[j] = thetas[j] - (alpha / len(X)) * np.sum((hypothesis(X, thetas) - Y) * np.array(X[:, j])) + (0.1 * thetas[j] ** 2)
    thetas = next_thetas
 
print('Thetas: ' + str(thetas))
print('-----')

pred = [hypothesis(X, thetas) >= 0.5]
print('Accruracy is: ' + str(np.mean(pred == Y.flatten()) * 100) + '%')

u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u), len(v)))

# X = X.transpose()

# def mapFeatureForPlotting(X1, X2):
#     degree = 6
#     out = np.ones(1)
#     for i in range(1, degree+1):
#         for j in range(i+1):
#             out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))))
#     return out

# for i in range(len(u)):
#     for j in range(len(v)):
#         z[i,j] = np.dot(mapFeatureForPlotting(u[i], v[j]), thetas)

# mask = Y.flatten() == 1
# # X = data.iloc[:,:-1]
# passed = pl.scatter(X[mask][0], X[mask][1])
# failed = pl.scatter(X[~mask][0], X[~mask][1])
# pl.contour(u,v,z,0)
# pl.xlabel('Microchip Test1')
# pl.ylabel('Microchip Test2')
# pl.legend((passed, failed), ('Passed', 'Failed'))

# pl.show()
