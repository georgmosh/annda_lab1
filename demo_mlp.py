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
    means_B[1] -= 0.1

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
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training vs Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_accuracy(acc_train, acc_val, num_epochs):
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, acc_train, 'g', label='Training accuracy')
    plt.plot(epochs, acc_val, 'b', label='validation accuracy')
    plt.title('Training vs Validation accuracy')
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
    O1 = np.ones([o1.shape[0] + 1, o1.shape[1]])
    O1[:-1, :] = o1
    o1 = O1
    h2 = np.matmul(W2, o1)
    o2 = activation(h2)
    
    return h1, o1, h2, o2

def backward_pass(lr, x, h1, o1, h2, o2, t, W1, W2):
    # hidden layer
    delta_hidden = np.multiply((o2[0] - t), activation_der(o2[0]))
    dW2 = -lr * np.matmul(delta_hidden, np.transpose(o1))
    
    # output layer
    delta_hidden = delta_hidden.reshape(len(delta_hidden), 1)
    # delta_output = np.multiply(np.matmul(np.transpose(W2).shape,  np.transpose(delta_hidden).shape), activation_der(o1))
    delta_output = np.multiply(np.matmul(np.transpose(W2),  np.transpose(delta_hidden)), activation_der(o1))
    dW1 = -lr * np.matmul(delta_output, x)

    # weight update
    W2[0] = W2[0] + dW2
    W1 = W1 + dW1[:len(dW1)-1]

    return W1, W2

def MSE(t, o2):
    loss = np.dot((t - o2), (t - o2)) / t.shape[0]
    return loss

def mlp(hidden_nodes):
    num_epochs = 200
    lr = 0.01
    num_of_datapoints, class_A, class_B = generateDatapoints()
    x_axis = [class_A[0], class_B[0]]
    y_axis = [class_A[1], class_B[1]]
    losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    accuracies = np.zeros(num_epochs)
    val_accuracies = np.zeros(num_epochs)
    num_crossval_runs = 1
    
    class_A_all = [class_A[0], class_A[1], -np.ones(int(num_of_datapoints/2))]
    class_B_all = [class_B[0], class_B[1], np.ones(num_of_datapoints)]

    plot(class_A[0], class_A[1], class_B[0], class_B[1])
    dataset = np.transpose(np.concatenate((class_A_all, class_B_all), axis=1))
    np.random.shuffle(dataset)
    target_whole = dataset[:,2].copy()
    dataset[:,2] = 1

    d_train = int(0.8 * 1.5 * num_of_datapoints)
    data = dataset[0:d_train,:]
    val_data = dataset[d_train:,:]
    target = target_whole[0:d_train]
    val_target = target_whole[d_train:]
    
    for _ in range(0, num_crossval_runs):
        W1 = np.random.rand(hidden_nodes - 1, 3)
        W2 = np.random.rand(1, hidden_nodes)
        
        for epoch in range(0, num_epochs):
            ### --------------- TRAINING --------------- ##
            h1, o1, h2, o2 = forward_pass(data, W1, W2)
        
            loss = MSE(target, o2[0])
            losses[epoch] += loss

            accuracy = np.sum(np.abs(o2 - target) < 1) / target.shape[0]
            accuracies[epoch] += accuracy

            W1, W2 = backward_pass(lr, data, h1, o1, h2, o2, target, W1, W2)

            ### --------------- VALIDATION --------------- ##
            h1, o1, h2, o2 = forward_pass(val_data, W1, W2)

            loss = MSE(val_target, o2[0])
            val_losses[epoch] += loss

            accuracy = np.sum(np.abs(o2 - val_target) < 1) / val_target.shape[0]
            val_accuracies[epoch] += accuracy
    
    losses /= num_crossval_runs
    val_losses /= num_crossval_runs
    accuracies /= num_crossval_runs
    val_accuracies /= num_crossval_runs
    losses = np.ndarray.tolist(losses)
    val_losses = np.ndarray.tolist(val_losses)
    accuracies = np.ndarray.tolist(accuracies)
    val_accuracies = np.ndarray.tolist(val_accuracies)

    ### --------------- EVALUATION --------------- ##
    print("Average loss per epoch", losses, val_losses)
    plot_loss(losses, val_losses, num_epochs)

    print("Average accuracy per epoch", accuracies, val_accuracies)
    plot_accuracy(accuracies, val_accuracies, num_epochs)

mlp(3)