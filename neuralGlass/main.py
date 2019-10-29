import numpy as np
import matplotlib.pyplot as plt
import pandas

np.random.seed(42)
data = pandas.read_csv('glass.csv')

data = data.sample(frac=1).reset_index(drop=True)

labels = data['Type']
data = data.drop('Type', axis = 1)

data = data.to_numpy()
labels = labels.to_numpy()

one_hot_labels = np.zeros((214, 9))

for i in range(214):
    one_hot_labels[i, labels[i]] = 1

# np.random.shuffle(data)

X = data

x = X

X = np.array([[(X[i][j]-np.mean(X[:, j]))/(np.max(X[:, j])-np.min(X[:, j])) for j in range(len(X[i]))] for i in range(len(X))])

# X = np.array([np.insert(i, 0, 1) for i in X])



# plt.figure(figsize=(10,7))
# plt.scatter(X[:,0], X[:,1], c=labels, cmap='plasma', s=100, alpha=0.5)
# plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

instances = X.shape[0]
attributes = X.shape[1]
hidden_nodes = 9
output_labels = 9

wh = np.random.rand(attributes,hidden_nodes)
bh = np.random.randn(hidden_nodes)

wo = np.random.rand(hidden_nodes,output_labels)
bo = np.random.randn(output_labels)
lr = 0.01

error_cost = []

for epoch in range(15000):
############# feedforward

    # Phase 1
    zh = np.dot(X, wh) + bh
    ah = sigmoid(zh)

    # Phase 2
    zo = np.dot(ah, wo) + bo
    ao = softmax(zo)

########## Back Propagation

########## Phase 1

    dcost_dzo = ao - one_hot_labels
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

    dcost_bo = dcost_dzo

########## Phases 2

    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = sigmoid_der(zh)
    dzh_dwh = X
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    dcost_bh = dcost_dah * dah_dzh

    # Update Weights ================

    wh -= lr * dcost_wh
    bh -= lr * dcost_bh.sum(axis=0)

    wo -= lr * dcost_wo
    bo -= lr * dcost_bo.sum(axis=0)

    if epoch % 200 == 0:
        loss = np.sum(-one_hot_labels * np.log(ao))
        print('Loss function value: ', loss)
        error_cost.append(loss)
    # print("a")