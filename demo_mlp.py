import numpy as np
import matplotlib.pyplot as plt
import random
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
    plt.ylabel('Accuracy')
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

def backward_pass(lr, x, h1, o1, h2, o2, t, W1, W2, dW1, dW2, alpha):
    # hidden layer
    delta_hidden = np.multiply((o2[0] - t), activation_der(o2[0]))
    theta = (1 - alpha) * np.matmul(delta_hidden, np.transpose(o1))
    dW2 = alpha * dW2 -lr * theta

    # output layer
    delta_hidden = delta_hidden.reshape(len(delta_hidden), 1)
    # delta_output = np.multiply(np.matmul(np.transpose(W2).shape,  np.transpose(delta_hidden).shape), activation_der(o1))
    delta_output = np.multiply(np.matmul(np.transpose(W2), np.transpose(delta_hidden)), activation_der(o1))
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
    num_of_datapoints, class_A, class_B = generateDatapoints()
    x_axis = [class_A[0], class_B[0]]
    y_axis = [class_A[1], class_B[1]]
    losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    accuracies = np.zeros(num_epochs)
    val_accuracies = np.zeros(num_epochs)
    momentum_coef = 0.9
    training_percentage = 0.8
    num_crossval_runs = 1
    batch_mode = 1
    
    class_A_all = [class_A[0], class_A[1], -np.ones(int(num_of_datapoints/2))]
    class_B_all = [class_B[0], class_B[1], np.ones(num_of_datapoints)]

    plot(class_A[0], class_A[1], class_B[0], class_B[1])
    dataset = np.transpose(np.concatenate((class_A_all, class_B_all), axis=1))
    np.random.shuffle(dataset)
    target_whole = dataset[:,2].copy()
    dataset[:,2] = 1

    d_train = int(training_percentage * 1.5 * num_of_datapoints)
    data = dataset[0:d_train,:]
    val_data = dataset[d_train:,:]
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

            if(batch_mode == 1):
                loss = MSE(target, o2[0])
                losses[epoch] += loss
            else:
                loss = 0
                for d in range(0, int(1.5 * training_percentage * num_of_datapoints)):
                    point_loss = MSE2(target[d], o2[0][d], target.shape[0])
                    loss += point_loss
                losses[epoch] += loss

            accuracy = np.sum(np.abs(o2 - target) < 1) / target.shape[0]
            accuracies[epoch] += accuracy

            W1, W2, dW1, dW2 = backward_pass(lr, data, h1, o1, h2, o2, target, W1, W2, dW1, dW2, momentum_coef)

            ### --------------- VALIDATION --------------- ##
            h1, o1, h2, o2 = forward_pass(val_data, W1, W2)

            if (batch_mode == 1):
                loss = MSE(val_target, o2[0])
                val_losses[epoch] += loss
            else:
                loss = 0
                for d in range(0, int(1.5 * (1-training_percentage) * num_of_datapoints)):
                    point_loss = MSE2(val_target[d], o2[0][d], val_target.shape[0])
                    loss += point_loss
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


def mlp_less1(hidden_nodes):
    num_epochs = 250
    lr = 0.01
    num_of_datapoints, class_A, class_B = generateDatapoints()
    x_axis = [class_A[0], class_B[0]]
    y_axis = [class_A[1], class_B[1]]
    losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    accuracies = np.zeros(num_epochs)
    val_accuracies = np.zeros(num_epochs)
    momentum_coef = 0.9
    training_percentage = 0.8
    num_crossval_runs = 1
    batch_mode = 1

    class_A_all = [class_A[0], class_A[1], -np.ones(int(num_of_datapoints / 2))]
    class_B_all = [class_B[0], class_B[1], np.ones(num_of_datapoints)]

    # plot(class_A[0], class_A[1], class_B[0], class_B[1])
    data_whole = np.transpose(np.concatenate((class_A_all, class_B_all), axis=1))

    startA = random.randrange(0, int(num_of_datapoints / 2) - int(num_of_datapoints / 8))
    newvalA = data_whole[startA:startA + int(num_of_datapoints / 8),:]
    target_newvalA = newvalA[:,2].copy()
    newvalA[:,2] = 1

    startB = int(num_of_datapoints / 2) + random.randrange(0, int(num_of_datapoints) - int(num_of_datapoints / 4))
    newvalB = data_whole[startB:startB + int(num_of_datapoints / 4),:]
    target_newvalB = newvalB[:, 2].copy()
    newvalB[:, 2] = 1

    newD1 = data_whole[0:startA, :]
    newD2 = data_whole[startA + int(num_of_datapoints / 8):startB,:]
    newD3 = data_whole[startB + int(num_of_datapoints / 4):,:]
    dataset = np.concatenate((newD1, newD2, newD3), axis=0)

    np.random.shuffle(dataset)
    target_whole = dataset[:, 2].copy()
    dataset[:, 2] = 1

    d_train = int(training_percentage * dataset.shape[0])
    data = dataset[0:d_train, :]
    val_init = dataset[d_train:, :]
    target = target_whole[0:d_train]
    val_target_init = target_whole[d_train:]

    val_data = np.concatenate((val_init, newvalA, newvalB), axis=0)
    val_target = np.concatenate((val_target_init, target_newvalA, target_newvalB), axis=0)

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
                for d in range(0, int(data.shape[0])):
                    point_loss = MSE2(target[d], o2[0][d], target.shape[0])
                    loss += point_loss
                losses[epoch] += loss

            accuracy = np.sum(np.abs(o2 - target) < 1) / target.shape[0]
            accuracies[epoch] += accuracy

            W1, W2, dW1, dW2 = backward_pass(lr, data, h1, o1, h2, o2, target, W1, W2, dW1, dW2, momentum_coef)

            ### --------------- VALIDATION --------------- ##
            h1, o1, h2, o2 = forward_pass(val_data, W1, W2)

            if (batch_mode == 1):
                loss = MSE(val_target, o2[0])
                val_losses[epoch] += loss
            else:
                loss = 0
                for d in range(0, int(val_data.shape[0])):
                    point_loss = MSE2(val_target[d], o2[0][d], val_target.shape[0])
                    loss += point_loss
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

def mlp_less2(hidden_nodes):
    num_epochs = 250
    lr = 0.01
    num_of_datapoints, class_A, class_B = generateDatapoints()
    x_axis = [class_A[0], class_B[0]]
    y_axis = [class_A[1], class_B[1]]
    losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    accuracies = np.zeros(num_epochs)
    val_accuracies = np.zeros(num_epochs)
    momentum_coef = 0.9
    training_percentage = 0.8
    num_crossval_runs = 1
    batch_mode = 1

    class_A_all = [class_A[0], class_A[1], -np.ones(int(num_of_datapoints / 2))]
    class_B_all = [class_B[0], class_B[1], np.ones(num_of_datapoints)]

    # plot(class_A[0], class_A[1], class_B[0], class_B[1])
    data_whole = np.transpose(np.concatenate((class_A_all, class_B_all), axis=1))

    startA = random.randrange(0, int(num_of_datapoints / 2) - int(num_of_datapoints / 4))
    newvalA = data_whole[startA:startA + int(num_of_datapoints / 4),:]
    target_newvalA = newvalA[:,2].copy()
    newvalA[:,2] = 1

    newD1 = data_whole[0:startA, :]
    newD2 = data_whole[startA + int(num_of_datapoints / 4):,:]
    dataset = np.concatenate((newD1, newD2), axis=0)

    np.random.shuffle(dataset)
    target_whole = dataset[:, 2].copy()
    dataset[:, 2] = 1

    d_train = int(training_percentage * dataset.shape[0])
    data = dataset[0:d_train, :]
    val_init = dataset[d_train:, :]
    target = target_whole[0:d_train]
    val_target_init = target_whole[d_train:]

    val_data = np.concatenate((val_init, newvalA), axis=0)
    val_target = np.concatenate((val_target_init, target_newvalA), axis=0)

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
                for d in range(0, int(data.shape[0])):
                    point_loss = MSE2(target[d], o2[0][d], target.shape[0])
                    loss += point_loss
                losses[epoch] += loss

            accuracy = np.sum(np.abs(o2 - target) < 1) / target.shape[0]
            accuracies[epoch] += accuracy

            W1, W2, dW1, dW2 = backward_pass(lr, data, h1, o1, h2, o2, target, W1, W2, dW1, dW2, momentum_coef)

            ### --------------- VALIDATION --------------- ##
            h1, o1, h2, o2 = forward_pass(val_data, W1, W2)

            if (batch_mode == 1):
                loss = MSE(val_target, o2[0])
                val_losses[epoch] += loss
            else:
                loss = 0
                for d in range(0, int(val_data.shape[0])):
                    point_loss = MSE2(val_target[d], o2[0][d], val_target.shape[0])
                    loss += point_loss
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

def mlp_less3(hidden_nodes):
    num_epochs = 250
    lr = 0.01
    num_of_datapoints, class_A, class_B = generateDatapoints()
    x_axis = [class_A[0], class_B[0]]
    y_axis = [class_A[1], class_B[1]]
    losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    accuracies = np.zeros(num_epochs)
    val_accuracies = np.zeros(num_epochs)
    momentum_coef = 0.9
    training_percentage = 0.8
    num_crossval_runs = 1
    batch_mode = 1

    class_A_all = [class_A[0], class_A[1], -np.ones(int(num_of_datapoints / 2))]
    class_B_all = [class_B[0], class_B[1], np.ones(num_of_datapoints)]

    # plot(class_A[0], class_A[1], class_B[0], class_B[1])
    data_whole = np.transpose(np.concatenate((class_A_all, class_B_all), axis=1))

    startB = int(num_of_datapoints / 2) + random.randrange(0, int(num_of_datapoints) - int(num_of_datapoints / 2))
    newvalB = data_whole[startB:startB + int(num_of_datapoints / 2), :]
    target_newvalB = newvalB[:, 2].copy()
    newvalB[:, 2] = 1

    newD1 = data_whole[0:startB, :]
    newD2 = data_whole[startB + int(num_of_datapoints / 2):, :]
    dataset = np.concatenate((newD1, newD2), axis=0)

    np.random.shuffle(dataset)
    target_whole = dataset[:, 2].copy()
    dataset[:, 2] = 1

    d_train = int(training_percentage * dataset.shape[0])
    data = dataset[0:d_train, :]
    val_init = dataset[d_train:, :]
    target = target_whole[0:d_train]
    val_target_init = target_whole[d_train:]

    val_data = np.concatenate((val_init, newvalB), axis=0)
    val_target = np.concatenate((val_target_init, target_newvalB), axis=0)

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
                for d in range(0, int(data.shape[0])):
                    point_loss = MSE2(target[d], o2[0][d], target.shape[0])
                    loss += point_loss
                losses[epoch] += loss

            accuracy = np.sum(np.abs(o2 - target) < 1) / target.shape[0]
            accuracies[epoch] += accuracy

            W1, W2, dW1, dW2 = backward_pass(lr, data, h1, o1, h2, o2, target, W1, W2, dW1, dW2, momentum_coef)

            ### --------------- VALIDATION --------------- ##
            h1, o1, h2, o2 = forward_pass(val_data, W1, W2)

            if (batch_mode == 1):
                loss = MSE(val_target, o2[0])
                val_losses[epoch] += loss
            else:
                loss = 0
                for d in range(0, int(val_data.shape[0])):
                    point_loss = MSE2(val_target[d], o2[0][d], val_target.shape[0])
                    loss += point_loss
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

mlp_less3(3)