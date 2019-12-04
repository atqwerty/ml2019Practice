from scipy.io import loadmat
from statistics import mean
import numpy as np
import matplotlib.pyplot as pl

def cost(X, y, thetas):
    h = X.dot(thetas)
    return np.sum(np.square(h - y)) / (2 * len(X)) + np.sum(np.square(thetas[1:])) * 1/(2*len(X))

def gradient_descent(m, X, y, thetas):
    costs = []
    next_thetas = np.ones(2).astype(float)
    for j in range(2):
        next_thetas[j] = thetas[j] - (1 / m) * np.sum((X.dot(thetas) - y) * np.array(X[:, j]))
    thetas = next_thetas

    costs.append(cost(X, y, thetas))
    return thetas, costs

# 1.1
data = loadmat('ex5data1.mat')

# 1.2
X = data['X']
y = data['y']

X_show = X
pl.scatter(X_show, y, marker="x", c="red")
pl.show()

X = np.array([np.insert(i, 0, 1) for i in X])

thetas = np.ones(2)

print(cost(X, y, thetas))

# 1.3
thetas, _ = gradient_descent(len(X), X, y, thetas)
print(thetas)

# 1.4
def best_fit(X, Y):
    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar
    return a, b

a, b = best_fit(X_show, y)
pl.scatter(X_show, y, marker="x", c="red")
yfit = [a + b * xi for xi in X_show]
pl.plot(X_show, yfit)
pl.show()

# 2.1
X_val = data['Xval']
y_val = data['yval']
X_val = np.array([np.insert(i, 0, 1) for i in X_val])
thetas_val = np.ones(2)

pl.plot(np.arange(1, m+1), error_train, label="Train")
pl.plot(np.arange(1, m+1), error_val, label="Cross Validation")
pl.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
pl.title('Learning curve for linear regression')
pl.xlabel('Number of training examples')
pl.ylabel('Error')
pl.show()
