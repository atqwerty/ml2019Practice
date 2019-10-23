import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import pandas
# import cv2
# import tkinter
from copy import deepcopy

# matplotlib.use('TkAgg')

def isclose(a, b, rtol=1e-05, atol=1e-08):
    return np.abs(a-b) <= (atol + rtol * np.abs(b))

def sigma(z):
	return 1/(1 + np.exp(-z))

data = pandas.read_csv('glass.csv')

data = data.sample(frac=1).reset_index(drop=True)

y_data = data['Type']
data = data.drop('Type', axis = 1)

data = data.to_numpy()
y_data = y_data.to_numpy()

x = data

# y = [(i-np.mean(y))/(np.max(y)-np.min(y)) for i in y]

# x = np.array([[(x[i][j]-np.mean(x[:, j]))/(np.max(x[:, j])-np.min(x[:, j])) for j in range(len(x[i]))] for i in range(len(x))])

y = deepcopy(y_data)

x = np.array([np.insert(i, 0, 1) for i in x])
m = len(x)

# theta = np.ndarray(shape=(7, 10))
# theta.fill(0)
theta = np.random.randn(7, 10) * 0.001

lr = 0.001
alpha = 0.7
dw_s=0

costs = np.zeros((7, 10000))

def cost_function(X, y, thetas):
    h = 1/(1 + np.exp(-X.dot(thetas)))
    cost_1 = np.log(h)
    cost_2 = np.array([np.log(1-i) for i in h])
    summ = y.dot(cost_1) + np.array([1-i for i in y]).dot(cost_2)
    return -summ/m

for type in range(7):  
        y = deepcopy(y_data)  
        idxs_i = y == type + 1
        y[idxs_i] = 1
        y[~idxs_i] = 0
        for i in range(10000):
                Z = x.dot(theta[type])
                a = sigma(Z)
                dz = a - y
                dw = x.T.dot(dz)/m
                dw += (np.sum(theta[type]) - theta[type][0]) / m * alpha
                # dw_s = dw_s * 0.9 + 0.1 * dw
                theta[type] -= lr * dw
                costs[type][i] = cost_function(x, y, theta[type])
# costs.append()


for type in range(7):
        print(theta[type])
        plt.plot(range(len(costs[type])), costs[type])
        # plt.axis([xmin, xmax, ymin, ymax])
        plt.title("The cost function plot of glass of the following type " + str(type))
        plt.show()
        # input("")
        # exit(0);


rres = 0.0
for type in range(7):
    for i in range(len(y_data)):        
        y[i] = 1 if y_data[i] == (type + 1) else 0
    # print(y)
    pred = [sigma(np.dot(x, theta[type])) >= 0.5]
    print(np.mean(pred == y.flatten()) * 100)
    rres += np.mean(pred == y.flatten()) * 100


# exit(0)

print("")

result = 0.0
for i in range(len(x)):
    res = []
    for type in range(7):
        res.append(sigma(np.dot(x[i], theta[type])))        
#     print(res)
#     print (np.argmax(res)+1)
#     print (y_data[i])
    result += np.argmax(res) + 1 == y_data[i]

print (rres / 7)
    
print (result / m * 100) 