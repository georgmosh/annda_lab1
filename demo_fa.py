import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

def generateDatapoints():
    np.random.seed(0)
    
    num_of_datapoints = 441

    data = []
    x_axis = []
    y_axis = []
    expectation = []
    
    for i in np.arange(-5, 5.5, 0.5):
        for j in np.arange(-5, 5.5, 0.5):
            f = np.exp(-0.1 * np.dot(i, i)) * np.exp(-0.1 * np.dot(j, j))
            elem = np.zeros(3)
            elem[0] += i
            elem[1] += j
            elem[2] += f
            data.append(elem)
            x_axis.append(i)
            y_axis.append(j)
            expectation.append(f)
            
    data = np.array(data)
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)
    expectation = np.array(expectation)

    return num_of_datapoints, data, x_axis, y_axis, expectation
     
def plot_loss(loss_train, loss_val, num_epochs):
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
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
    num_epochs = 50
    num_classes = 2
    lr = 0.3
    num_of_datapoints, dataset, x_axis, y_axis, expectation = generateDatapoints()
    losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    num_crossval_runs = 5
    
    np.random.shuffle(dataset)
    target_whole = dataset[:,2].copy()
    dataset[:,2] = 1
    
    d_train = int(0.8 * num_of_datapoints)
    data = dataset[0:d_train,:] 
    val_data = dataset[d_train:,:]
    target = target_whole[0:d_train]
    val_target = target_whole[d_train:]
    
    for _ in range(0, num_crossval_runs):
        W1 = np.random.rand(hidden_nodes, 3)
        W2 = np.random.rand(1, hidden_nodes)
        
        for epoch in range(0, num_epochs):
            ### --------------- TRAINING --------------- ##
            h1, o1, h2, o2 = forward_pass(data, W1, W2)
        
            loss = MSE(target, o2[0])
            losses[epoch] += loss
        
            backward_pass(lr, data, h1, o1, h2, o2, target, W1, W2)
    
            ### --------------- VALIDATION --------------- ##
            h1, o1, h2, o2 = forward_pass(val_data, W1, W2)
        
            loss = MSE(val_target, o2[0])
            val_losses[epoch] += loss
    
    losses /= num_crossval_runs
    val_losses /= num_crossval_runs
    losses = np.ndarray.tolist(losses)
    val_losses = np.ndarray.tolist(val_losses)
    
    ### --------------- EVALUATION --------------- ##
    print("Average loss per epoch", losses, val_losses)
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