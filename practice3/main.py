import csv
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

def readRawData():
    data = []

    with open('train_data.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            data.append(row)

    csvFile.close()
    return data

def convertData(rawData):
    rawData.pop(0)
    for i in range(len(rawData)):
        for j in range(len(rawData[0])):
            rawData[i][j] = float(rawData[i][j])

    return np.transpose(rawData)

def normalize(arr):
    minVal = np.amin(arr)
    maxVal = np.amax(arr)
    for i in range(len(arr)):
        arr[i] = (arr[i] - minVal) / (maxVal - minVal)

    return arr


rawData = readRawData()
convertedData = convertData(rawData)

X = np.array([convertedData[0], convertedData[1], convertedData[2], convertedData[3]])
Y = np.array(convertedData[4])

for i in range(4):
    X[i] = normalize(X[i])

Y = normalize(Y)

print(X)