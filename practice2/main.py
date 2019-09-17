import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]

def costFunction(thetaZero, thetaOne, thetaTwo, xZero, xOne, y):
    return (sum(((thetaZero + thetaOne * xZero + thetaTwo * xOne) - y) ** 2) / (2 * 47))

def gradientDescent(xZero, xOne, y, alpha, thetaZero, thetaOne, thetaTwo, epochs, n):
    for smt in range (epochs):
        gradienZero = 0.0
        gradientOne = 0.0
        gradientTwo = 0.0
        for i in range(47):
            gradienZero += (1/n) * ((thetaTwo * xOne[i] + thetaOne * xZero[i] + thetaZero) - y[i])
            gradientOne += (1/n) * xZero[i] * ((thetaZero + thetaOne * xZero[i] + thetaTwo * xOne[i]) - y[i])
            gradientTwo += (1/n) * xOne[i] * ((thetaZero + thetaOne * xZero[i] + thetaTwo * xOne[i]) - y[i])

        thetaZero = thetaZero - (alpha * gradienZero)
        thetaOne = thetaOne - (alpha * gradientOne)
        thetaTwo = thetaTwo - (alpha * gradientTwo)

    print(thetaZero)
    print(thetaOne)
    print(thetaTwo)
    return [costFunction(thetaZero, thetaOne, thetaTwo, xZero, xOne, y), thetaZero, thetaOne, thetaTwo]


# 2d array of floats from file
data = np.loadtxt(fname = 'ex1data2.txt', delimiter = ",", usecols=[0, 1, 2])

# just empty arrays as x and y
xZero = np.zeros([47, 1])
xOne = np.zeros([47, 1])
y = np.zeros([47, 1])
# holder = np.ones([47, 1])

# im stoopid
counter = 0

# divide it into x's and y's
for row in data:
    xZero[counter] = row[0]
    xOne[counter] = row[1]
    y[counter] = row[2]
    counter += 1

minXZero = np.amin(xZero)
minXOne = np.amin(xOne)
minY = np.amin(y)
maxXZero = np.amax(xZero)
maxXOne = np.amax(xOne)
maxY = np.amax(y)

for i in range(47):
    xZero[i] = (xZero[i] - minXZero) / (maxXZero - minXZero)
    xOne[i] = (xOne[i] - minXOne) / (maxXOne - minXOne)
    y[i] = (y[i] - minY) / (maxY - minY)

n, m = np.shape(xZero)
epoches = 1500
alpha = 0.01
thetaZero = 0.0
thetaOne = 0.0
thetaTwo = 0.0

print("Cost function before: %f" % costFunction(thetaZero, thetaOne, thetaTwo, xZero, xOne, y))

answer = gradientDescent(xZero, xOne, y, alpha, thetaZero, thetaOne, thetaTwo, epoches, n)

gradAnswer = answer[0]
thetaZero = answer[1]
thetaOne = answer[2]
thetaTwo = answer[3]

print("Cost function after: %f" % gradAnswer)

ax = pl.figure().gca(projection="3d")

ax.scatter(xZero, xOne, y)

point  = np.array([0.0, 0.0, thetaZero])
normal = np.array(cross([1, 0, thetaTwo], [0, 1, thetaOne]))
xx, yy = np.meshgrid([0, 1], [0, 1])
z = (-normal[0] * xx - normal[1] * yy + point.dot(normal)) * 1. / normal[2]
ax.plot_surface(xx, yy, z, alpha=0.2, color="r")

pl.show()