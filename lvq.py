import numpy as np
import math
import time

def error(i, w):
    return math.sqrt(np.sum(np.subtract(i,w)**2))

train = np.array([[1,3,0], [3,4,0], [6,1,1], [8,3,1], [9,1,1], [1,6,0]], dtype=np.float64)
weights = np.array([[3,1],[7,4]], dtype=np.float64)
learningRate = 0.5

start = time()
for input in train:
    minimVal = float('inf')
    minimInx = -1
    for i,w in enumerate(weights):
        tempVal = error(input[:2], w)
        if  tempVal < minimVal :
            minimVal = tempVal
            minimInx = i
    if input[2]==minimInx:
        weights[minimInx] = weights[minimInx] + learningRate*(input[:2] - weights[minimInx])
    else:
        weights[minimInx] = weights[minimInx] - learningRate*(input[:2] - weights[minimInx])
    #learningRate = 0.1*learningRate

spent = time() - start
print spent
print weights
