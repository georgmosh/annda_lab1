import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

from mpl_toolkits.mplot3d import Axes3D


def generateDatapoints():
    np.random.seed(72)

    num_of_datapoints = 441

    data = []
    x_coord = []
    y_coord = []
    z_coord = []

    for i in np.arange(-5, 5.5, 0.5):
        for j in np.arange(-5, 5.5, 0.5):
            f = np.exp(-0.1 * np.dot(i, i)) * np.exp(-0.1 * np.dot(j, j)) - 0.5
            elem = np.zeros(3)
            elem[0] += i
            elem[1] += j
            elem[2] += f
            data.append(elem)
            x_coord.append(i)
            y_coord.append(j)
            z_coord.append(f)

    data = np.array(data)
    x_coord = np.array(x_coord)
    y_coord = np.array(y_coord)
    z_coord = np.array(z_coord)

    plot_Gaussian(x_coord, y_coord, z_coord)

    return num_of_datapoints, data

def plot_Gaussian(x, y, z):
    fig = plt.figure()
    axis = Axes3D(fig)
    axis.scatter(xs = x, ys = y, zs = z, zdir='z')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_loss(loss_train, loss_val, num_epochs):
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training vs Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def activation2(x):
    out = 1 / (1 + np.exp(-x)) - 0.5
    return out

def activation_der2(x):
    out = (1 + x) * (1 - x)
    return out

def forward_pass(x, W1, W2):
    # input layer
    h1 = np.matmul(W1, np.transpose(x))
    o1 = activation2(h1)

    # hidden layer
    O1 = np.ones([o1.shape[0] + 1, o1.shape[1]])
    O1[:-1, :] = o1
    o1 = O1
    h2 = np.matmul(W2, o1)
    o2 = activation2(h2)

    return h1, o1, h2, o2

def backward_pass(lr, x, h1, o1, h2, o2, t, W1, W2, dW1, dW2, alpha):
    # hidden layer
    delta_hidden = np.multiply((o2[0] - t), activation_der2(o2[0]))
    theta = (1 - alpha) * np.matmul(delta_hidden, np.transpose(o1))
    dW2 = alpha * dW2 -lr * theta

    # output layer
    delta_hidden = delta_hidden.reshape(len(delta_hidden), 1)
    # delta_output = np.multiply(np.matmul(np.transpose(W2).shape,  np.transpose(delta_hidden).shape), activation_der(o1))
    delta_output = np.multiply(np.matmul(np.transpose(W2), np.transpose(delta_hidden)), activation_der2(o1))
    theta = (1 - alpha) * np.matmul(delta_output, x)
    dW1 = alpha * dW1 -lr * theta

    # weight update
    W2[0] = W2[0] + dW2
    W1 = W1 + dW1[:len(dW1) - 1]

    return W1, W2, dW1, dW2

def MSE(t, o2):
    loss = np.dot((t - o2), (t - o2)) / t.shape[0]
    return loss

def MSE2(t, o2, dim):
    loss = np.dot((t - o2), (t - o2)) / dim
    return loss

def mlp(hidden_nodes):
    num_epochs = 250
    lr = 0.01
    num_of_datapoints, dataset = generateDatapoints()
    losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    momentum_coef = 0.9
    training_percentage = 0.8
    num_crossval_runs = 1
    batch_mode = 1

    np.random.shuffle(dataset)
    target_whole = dataset[:, 2].copy()
    dataset[:, 2] = 1

    d_train = int(training_percentage * num_of_datapoints)
    data = dataset[0:d_train, :]
    val_data = dataset[d_train:, :]
    target = target_whole[0:d_train]
    val_target = target_whole[d_train:]

    for _ in range(0, num_crossval_runs):
        W1 = np.random.rand(hidden_nodes - 1, 3)
        W2 = np.random.rand(1, hidden_nodes)
        dW1 = 0
        dW2 = 0

        for epoch in range(0, num_epochs):
            ### --------------- TRAINING --------------- ##
            h1, o1, h2, o2 = forward_pass(data, W1, W2)

            if (batch_mode == 1):
                loss = MSE(target, o2[0])
                losses[epoch] += loss
            else:
                loss = 0
                for d in range(0, int(training_percentage * num_of_datapoints)):
                    point_loss = MSE2(target[d], o2[0][d], target.shape[0])
                    loss += point_loss
                losses[epoch] += loss

            W1, W2, dW1, dW2 = backward_pass(lr, data, h1, o1, h2, o2, target, W1, W2, dW1, dW2, momentum_coef)

            ### --------------- VALIDATION --------------- ##
            h1, o1, h2, o2 = forward_pass(val_data, W1, W2)

            if (batch_mode == 1):
                loss = MSE(val_target, o2[0])
                val_losses[epoch] += loss
            else:
                loss = 0
                for d in range(0, int((1 - training_percentage) * num_of_datapoints)):
                    point_loss = MSE2(val_target[d], o2[0][d], val_target.shape[0])
                    loss += point_loss
                val_losses[epoch] += loss

        ### --------------- TESTING --------------- ##
        h1, o1, h2, o2 = forward_pass(dataset, W1, W2)

        x_coord = dataset[:,0]
        y_coord = dataset[:,1]
        z_coord = o2[0]

        plot_Gaussian(x_coord, y_coord, z_coord)

    losses /= num_crossval_runs
    val_losses /= num_crossval_runs
    losses = np.ndarray.tolist(losses)
    val_losses = np.ndarray.tolist(val_losses)

    ### --------------- EVALUATION --------------- ##
    print("Average loss per epoch", losses, val_losses)
    plot_loss(losses, val_losses, num_epochs)

mlp(15)