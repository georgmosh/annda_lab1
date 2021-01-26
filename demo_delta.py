import torch
import numpy as np
import matplotlib.pyplot as plt


### 3.1.1

## setting seed
np.random.seed(0)

n = 100
mA = [2, 2]
mB = [-2, -2]
mA = [0.5, 1]
mB = [-1, 1]
sigmaA = 0.5
sigmaB = 0.5

## Class A
classA = np.zeros([3, n])
classA[0, :] = np.random.normal(0, 1, n) * sigmaA + mA[0]
classA[1, :] = np.random.normal(0, 1, n) * sigmaA + mA[1]
classA[2, :] = -1


## Class B
classB = np.zeros([3, n])
classB[0, :] = np.random.normal(0, 1, n) * sigmaB + mB[0]
classB[1, :] = np.random.normal(0, 1, n) * sigmaB + mB[1]
classB[2, :] = 1


# ## plot
# plt.scatter(classA[0, :], classA[1, :], color='red', label="class A")
# plt.scatter(classB[0, :], classB[1, :], color='blue', label="class B")

# plt.show(block=False)
# plt.pause(1)
# plt.close("all")

###
data = np.concatenate((classA, classB), axis=1).T
np.random.shuffle(data)
data = data.T
targets = data[2, :].copy()
data[2, :] = 1 ## bias

##
xmax = np.max(data[0, :])
xmin = np.min(data[0, :])

#3.1.2 perceptron
## initialize W
lr = 0.001
W = np.random.random(3)
# W = [0.1, 0.8, 5]
# W[2] = 0 # bias = 0

print("")


for epoch in range(10):

    dw = -lr*np.matmul((np.matmul(W, data)-targets),data.T)
    W += dw


# plot
# plot normal vector
Wn = W / np.linalg.norm(W)
### computing boundary
ymax = (- W[0] * xmax - W[2])/W[1]
ymin = (- W[0] * xmin - W[2])/W[1]
plt.scatter(classA[0, :], classA[1, :], color='red', label="class A")
plt.scatter(classB[0, :], classB[1, :], color='blue', label="class B")

plt.plot([0, W[0]], [0, W[1]])
plt.plot([xmin, xmax], [ymin, ymax])
plt.legend()
plt.axis('equal')
plt.show()

print("")