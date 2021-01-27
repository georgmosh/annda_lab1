import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

def generateDatapoints():
    np.random.seed(0)
    
    num_of_datapoints = 100

    means_A = np.ones(2)
    means_A[1] -= 0.7

    sigma_A = 0.2

    means_B = np.zeros(2)
    means_B -= 0.1

    sigma_B = 0.3

    class_A1 = np.random.normal(loc=0, scale=1, size=int(num_of_datapoints/2)) * sigma_A + means_A[0]
    class_A2 = np.random.normal(loc=0, scale=1, size=int(num_of_datapoints/2)) * sigma_A + means_A[1]
    class_A = [class_A1, class_A2]

    class_B1 = np.random.normal(loc=0, scale=1, size=num_of_datapoints) * sigma_B + means_B[0]
    class_B2 = np.random.normal(loc=0, scale=1, size=num_of_datapoints) * sigma_B + means_B[1]
    class_B = [class_B1, class_B2]

    return num_of_datapoints, class_A, class_B
     
def plot(X, y, X1, y1):
    plt.scatter(X, y, color='r')
    plt.scatter(X1, y1, color='g')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
def plot_with_boundary(X, y, X1, y1, p1, p2):
    plt.scatter(X, y, color='r')
    plt.scatter(X1, y1, color='g')
    drawLine2P(p1, p2, [-2.5,1.5])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
def drawLine2P(x,y,xlims):
    # Source:
    # https://stackoverflow.com/questions/9148927/matplotlib-extended-line-over-2-control-points
    xrange = np.arange(xlims[0],xlims[1],0.1)
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y)[0]
    plt.plot(xrange, k*xrange + b)
 
def plot_loss(loss_train, loss_val, num_epochs):
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    #plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()    
 
def activation(x):
    out = 2/(1 + np.exp(-x)) - 1
    return out

def activation_der(x):
    out = (1 + x) * (1 - x) / 2
    return out

def forward_pass(x, W1, W2):
    # input layer
    h1 = np.matmul(W1, np.transpose(x))
    o1 = activation(h1)
    
    # hidden layer
    h2 = np.matmul(W2, h1)
    o2 = activation(h2)
    
    return h1, o1, h2, o2

def backward_pass(lr, x, h1, o1, h2, o2, t, W1, W2):
    # hidden layer
    delta_hidden = np.multiply((o2[0] - t), activation_der(o2[0]))
    dW = -lr * np.matmul(delta_hidden, np.transpose(o1))
    W2[0] = W2[0] + dW
    
    # output layer
    delta_hidden = delta_hidden.reshape(len(delta_hidden), 1)
    delta_output = np.multiply(np.matmul(np.transpose(W2).shape,  np.transpose(delta_hidden).shape), activation_der(o1))
    dW = -lr * np.matmul(delta_output, x)
    W1 = W1 + dW

def MSE(t, o2):
    loss = 0.5 * np.dot((t - o2), (t - o2))
    return loss

def mlp(hidden_nodes):
    num_epochs = 10
    num_classes = 2
    lr = 0.3
    num_of_datapoints, class_A, class_B = generateDatapoints()
    x_axis = [class_A[0], class_B[0]]
    y_axis = [class_A[1], class_B[1]]
    losses = []
    
    class_A_all = [class_A[0], class_A[1], np.zeros(int(num_of_datapoints/2))]
    class_B_all = [class_B[0], class_B[1], np.ones(num_of_datapoints)]

#    plot(class_A[0], class_A[1], class_B[0], class_B[1])    
    data = np.transpose(np.concatenate((class_A_all, class_B_all), axis=1))
    np.random.shuffle(data)
    target = data[:,2].copy()
    data[:,2] = 1

    W1 = np.random.rand(hidden_nodes, 3)
    W2 = np.random.rand(1, hidden_nodes)
    
    for epoch in range(0, num_epochs):
        h1, o1, h2, o2 = forward_pass(data, W1, W2)
    
        loss = MSE(target, o2[0])
        losses.append(loss)
    
        backward_pass(lr, data, h1, o1, h2, o2, target, W1, W2)

    val_losses = []
    print("Loss per epoch", losses, val_losses)
    plot_loss(losses, val_losses, num_epochs)
    
# --------------------- garbage
#    for epoch in range(0, num_epochs):
#        delta = -lr * np.dot((np.dot(np.transpose(W), np.transpose(data)) - target), data)
#        W = W + delta

#    point_1 = [0, (-W[2]/W[0])]
#    point_2 = [(-W[2]/W[1]), 0]
#    plot_with_boundary(class_A[0], class_A[1], class_B[0], class_B[1], point_1, point_2)
#    print(W)

mlp(100)