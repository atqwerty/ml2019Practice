import nn
import pandas
import numpy as np

data = pandas.read_csv('glass.csv')

data = data.sample(frac=1).reset_index(drop=True)

Y = data['Type']
data = data.drop('Type', axis = 1)

data = data.to_numpy()
Y = Y.to_numpy()
y = []

for i in range(len(Y)):
    holder = np.zeros(6)
    if Y[i] == 1:
        holder[0] = 1
        
    if Y[i] == 2:
        holder[1] = 1
        
    if Y[i] == 3:
        holder[2] = 1
        
    if Y[i] == 5:
        holder[3] = 1
        
    if Y[i] == 6:
        holder[4] = 1
        
    if Y[i] == 7:
        holder[5] = 1
        
    y.append(holder)

y = np.asarray(y)


X = data

# x = X

X = np.array([[(X[i][j]-np.mean(X[:, j]))/(np.max(X[:, j])-np.min(X[:, j])) for j in range(len(X[i]))] for i in range(len(X))])

# X = np.array([np.insert(i, 0, 1) for i in X])

model = nn.NeuralNetwork()
model.init(X, y)

for i in range(1500):
    model.feed_forward()
    model.back_propogation()

print(model.output)