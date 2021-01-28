import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as F

def set_set(seed):
    np.random.seed(seed)


def generate_data(n, mA, mB, sigmaA, sigmaB, shuffle=False):

    ## Class A
    classA = np.zeros([3, n/2])
    classA[0, :] = np.random.normal(0, 1, n/2) * sigmaA + mA[0]
    classA[1, :] = np.random.normal(0, 1, n/2) * sigmaA + mA[1]
    classA[2, :] = -1 ## class A: negative class

    ## Class B
    classB = np.zeros([3, n])
    classB[0, :] = np.random.normal(0, 1, n) * sigmaB + mB[0]
    classB[1, :] = np.random.normal(0, 1, n) * sigmaB + mB[1]
    classB[2, :] = 1 ## class B: positive class

    data = np.concatenate((classA, classB), axis=1).T

    if shuffle:
        np.random.shuffle(data)
    data = data.T

    targets = data[2, :].copy()

    data[2, :] = 1  ## bias

    return targets, data, classA, classB

def generate_function(xmin,xmax,ymin,ymax,xstep,ystep, shuffle=False):

    ## meshgrid
    x = np.arange(xmin, xmax, xstep)
    x = np.repeat(x, x.shape[0])

    y = np.arange(ymin, ymax, ystep)
    y = np.tile(y, y.shape[0])

    targets = np.exp(-(x*x + y*y)/10)-0.5

    data = np.stack((x, y, targets), axis=0)

    if shuffle:
        np.random.shuffle(data.T)

    targets = data[2, :].copy()

    data[2, :] = np.ones(y.shape[0])


    return targets, data

def generate_time_series(N=1500, beta=0.2, gamma=0.1, n=10, tau=25):

    ## Generate Time series
    x = [1.5]
    for t in range(N-1):
        x_t = x[t]

        if len(x) <= tau:
            x_t_25 = 0
        else:
            x_t_25 = x[t-tau]
        x.append(x_t + beta*x_t_25/(1+x_t_25**n) - gamma*x_t)

    ### Split dataset
    x_test = x[-200:]

    ### 80/20 train/val
    idx = int(0.8 * (N-200-300))
    x_train = x[300:300+idx]
    x_val = x[300+idx:N-200]

    ### Constructing datasets
    data_train = []
    target_train = []
    for t in range(20, len(x_train)-5):
        data_train.append([x_train[t-20], x_train[t-15], x_train[t-10], x_train[t-5], x_train[t]] )
        target_train.append(x_train[t+5])
    ## shuffle ???
    data_train = np.asarray(data_train)
    target_train = np.asarray(target_train)

    # data_train_shuffle = np.zeros((data_train.shape[0], data_train.shape[1]+1))
    # data_train_shuffle[:, :5] = data_train
    # data_train_shuffle[:, -1] = target_train
    # np.random.shuffle(data_train_shuffle)
    # data_train = data_train_shuffle[:, :5]
    # target_train = data_train_shuffle[:, -1]

    data_val= []
    target_val = []
    for t in range(20, len(x_val)-5):
        data_val.append([x_val[t-20], x_val[t-15], x_val[t-10], x_val[t-5], x_val[t]] )
        target_val.append(x_val[t+5])
    ## shuffle ???
    data_val = np.asarray(data_val)
    target_val = np.asarray(target_val)

    data_test= []
    target_test = []
    for t in range(20, len(x_test)-5):
        data_test.append([x_test[t-20], x_test[t-15], x_test[t-10], x_test[t-5], x_test[t]] )
        target_test.append(x_test[t+5])
    ## shuffle ???
    data_test = np.asarray(data_test)
    target_test = np.asarray(target_test)

    return data_train, target_train, data_val, target_val, data_test, target_test








def generate_non_linear_data(n, mA, mB, sigmaA, sigmaB, shuffle=False):

    ## Class A
    classA = np.zeros([3, n])
    classA[0, :] = np.random.normal(0, 1, n) * sigmaA + mA[0]
    classA[1, :] = np.random.normal(0, 1, n) * sigmaA + mA[1]
    classA[2, :] = -1 ## class A: negative class

    ## Class B
    classB = np.zeros([3, n])
    classB[0, :] = np.random.normal(0, 1, n) * sigmaB + mB[0]
    classB[1, :] = np.random.normal(0, 1, n) * sigmaB + mB[1]
    classB[2, :] = 1 ## class B: positive class

    data = np.concatenate((classA, classB), axis=1).T

    if shuffle:
        np.random.shuffle(data)
    data = data.T

    targets = data[2, :].copy()

    data[2, :] = 1  ## bias

    return targets, data, classA, classB


class Perceptron:

    def __init__(self):
        ## generate data
        self.W = np.random.random(3)

    def plot_boundary(self, data, classA, classB, msg, currentD=None):

        ## setting the plot ranges
        xmax = np.max(data[0, :])
        xmax += np.sign(xmax) * 0.2 * xmax

        xmin = np.min(data[0, :])
        xmin -= np.sign(xmin) * 0.2 * xmin

        ymax = np.max(data[1, :])
        ymax += np.sign(ymax) * 0.2 * ymax

        ymin = np.min(data[1, :])
        ymin -= np.sign(ymin) * 0.2 * ymin

        ## plot normal vector
        Wn = self.W / np.linalg.norm(self.W)
        WnOrigin = np.zeros(Wn.shape)


        plt.figure()

        ### computing boundary
        if abs(self.W[0]/self.W[1]) <= 1:
            line_max = (- self.W[0] * xmax - self.W[2]) / self.W[1]
            line_min = (- self.W[0] * xmin - self.W[2]) / self.W[1]
            WnOrigin[0] = (xmin+xmax)/2
            WnOrigin[1] = (line_min+line_max)/2
            plt.plot([xmin, xmax], [line_min, line_max ])
            print()

        else:
            line_max = (- self.W[1] * ymin - self.W[2]) / self.W[0]
            line_min = (- self.W[1] * ymax - self.W[2]) / self.W[0]
            WnOrigin[0] = (line_min+line_max)/2
            WnOrigin[1] = (ymin+ymax)/2
            plt.plot([line_min, line_max], [ymax,ymin ])

        plt.scatter(classA[0, :], classA[1, :], color='red', label="class A")
        plt.scatter(classB[0, :], classB[1, :], color='blue', label="class B")

        plt.plot([WnOrigin[0], WnOrigin[0] + Wn[0]], [WnOrigin[1], WnOrigin[1] + Wn[1]])
        # plt.plot([0, Wn[0]], [0, Wn[1]])
        print("Wn:", Wn)


        if currentD is not None:
            plt.plot(currentD[0], currentD[1], "vy")

        plt.legend()
        plt.axis('equal')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title(msg)
        plt.show()
        print("")


    def train(self, targets, data, classA, classB, epochs=100, lr=0.3):


        for epoch in range(epochs):
            for index in range(data.shape[1]):
                currentD = data[:, index]


                target = targets[index]
                targetPred = np.matmul(self.W, currentD)

                self.plot_boundary(data, classA, classB, "Initial", currentD, )

                if np.sign(targetPred) == target:
                    pass  # do nothing ## correctly classfied
                    self.plot_boundary(data, classA, classB, "No Update",currentD,)
                    print("")
                    plt.close()
                    plt.close()

                else:
                    dw = (target * currentD)
                    self.W += dw
                    self.plot_boundary(data, classA, classB, "Updated",currentD,)
                    print("")
                    plt.close()
                    plt.close()
                print("")

class deltaRule:

    def __init__(self):
        ## generate data
        self.W = np.random.random(3)

    def plot_boundary(self, data, classA, classB, msg, currentD=None):

        ## setting the plot ranges
        xmax = np.max(data[0, :])
        xmax += np.sign(xmax) * 0.2 * xmax

        xmin = np.min(data[0, :])
        xmin -= np.sign(xmin) * 0.2 * xmin

        ymax = np.max(data[1, :])
        ymax += np.sign(ymax) * 0.2 * ymax

        ymin = np.min(data[1, :])
        ymin -= np.sign(ymin) * 0.2 * ymin

        ## plot normal vector
        Wn = self.W / np.linalg.norm(self.W)
        WnOrigin = np.zeros(Wn.shape)


        plt.figure()

        ### computing boundary
        if abs(self.W[0]/self.W[1]) <= 1:
            line_max = (- self.W[0] * xmax - self.W[2]) / self.W[1]
            line_min = (- self.W[0] * xmin - self.W[2]) / self.W[1]
            WnOrigin[0] = (xmin+xmax)/2
            WnOrigin[1] = (line_min+line_max)/2
            plt.plot([xmin, xmax], [line_min, line_max ])
            print()

        else:
            line_max = (- self.W[1] * ymin - self.W[2]) / self.W[0]
            line_min = (- self.W[1] * ymax - self.W[2]) / self.W[0]
            WnOrigin[0] = (line_min+line_max)/2
            WnOrigin[1] = (ymin+ymax)/2
            plt.plot([line_min, line_max], [ymax,ymin ])

        plt.scatter(classA[0, :], classA[1, :], color='red', label="class A")
        plt.scatter(classB[0, :], classB[1, :], color='blue', label="class B")

        plt.plot([WnOrigin[0], WnOrigin[0] + Wn[0]], [WnOrigin[1], WnOrigin[1] + Wn[1]])
        # plt.plot([0, Wn[0]], [0, Wn[1]])
        print("Wn:", Wn)


        if currentD is not None:
            plt.plot(currentD[0], currentD[1], "vy")

        plt.legend()
        plt.axis('equal')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title(msg)
        plt.show()
        print("")

    def train(self, targets, data, classA, classB, epochs=100, lr=0.3):

        self.plot_boundary(data, classA, classB, "Initial")
        for epoch in range(epochs):
            dw = -lr * np.matmul((np.matmul(self.W, data) - targets), data.T)
            self.W += dw

            self.plot_boundary(data, classA, classB, "No Update")


class deltaRuleSequential:

    def __init__(self):
        ## generate data
        self.W = np.random.random(3)


    def plot_boundary(self, data, classA, classB, msg, currentD=None):

        ## setting the plot ranges
        xmax = np.max(data[0, :])
        xmax += np.sign(xmax) * 0.2 * xmax

        xmin = np.min(data[0, :])
        xmin -= np.sign(xmin) * 0.2 * xmin

        ymax = np.max(data[1, :])
        ymax += np.sign(ymax) * 0.2 * ymax

        ymin = np.min(data[1, :])
        ymin -= np.sign(ymin) * 0.2 * ymin

        ## plot normal vector
        Wn = self.W / np.linalg.norm(self.W)
        WnOrigin = np.zeros(Wn.shape)


        plt.figure()

        ### computing boundary
        if abs(self.W[0]/self.W[1]) <= 1:
            line_max = (- self.W[0] * xmax - self.W[2]) / self.W[1]
            line_min = (- self.W[0] * xmin - self.W[2]) / self.W[1]
            WnOrigin[0] = (xmin+xmax)/2
            WnOrigin[1] = (line_min+line_max)/2
            plt.plot([xmin, xmax], [line_min, line_max ])
            print()

        else:
            line_max = (- self.W[1] * ymin - self.W[2]) / self.W[0]
            line_min = (- self.W[1] * ymax - self.W[2]) / self.W[0]
            WnOrigin[0] = (line_min+line_max)/2
            WnOrigin[1] = (ymin+ymax)/2
            plt.plot([line_min, line_max], [ymax,ymin ])

        plt.scatter(classA[0, :], classA[1, :], color='red', label="class A")
        plt.scatter(classB[0, :], classB[1, :], color='blue', label="class B")

        plt.plot([WnOrigin[0], WnOrigin[0] + Wn[0]], [WnOrigin[1], WnOrigin[1] + Wn[1]])
        # plt.plot([0, Wn[0]], [0, Wn[1]])
        print("Wn:", Wn)


        if currentD is not None:
            plt.plot(currentD[0], currentD[1], "vy")

        plt.legend()
        plt.axis('equal')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title(msg)
        plt.show()
        print("")

    def train(self, targets, data, classA, classB, epochs=100, lr=0.3):

        self.plot_boundary(data, classA, classB, "Initial")

        loss_seq = []
        n=data.shape[1]
        for epoch in range(epochs):
            for index in range(data.shape[1]):

                d = data[:, index]
                target = targets[index]
                dw = -lr * (np.matmul(self.W, d) - target) * d.T
                self.W += dw

                e = 0.5 * np.dot((targets - (np.matmul(self.W, data))), (targets - (np.matmul(self.W, data)))) / n

                loss_seq.append(e)

                # self.plot_boundary(data, classA, classB, "Initial")
        print("")

        self.plot_boundary(data, classA, classB, "Final")
        print("")


class two_layer_Perceptron:


    def __init__(self, inputNodes=3, outputNodes=1, hiddenNodes=2):
        ## generate data
        self.W = np.random.random([hiddenNodes-1, inputNodes])
        self.V = np.random.random([outputNodes, hiddenNodes])
        self.theta = 0
        self.psi = 0
        self.loss =[]
        self.acc = []
        self.val_loss =[]
        self.val_acc = []

    def activation(self, input):
        return 2/(1+np.exp(-input))-1

    def activation_derivative(self, input):
        return (1+input)*(1-input)/2

    def forward(self, input):
        hin = np.matmul(self.W, input)
        h = self.activation(hin)

        ## adding ones
        hout = np.ones([h.shape[0]+1, h.shape[1]])
        hout[:-1, :] = h

        oin = np.matmul(self.V, hout)
        output = self.activation(oin)
        return hout, output

    def backward(self, h, output, targets):

        deltaO = (output-targets) * self.activation_derivative(output)
        deltaH = np.matmul(self.V.T, deltaO) * self.activation_derivative(h)
        deltaH = deltaH[:-1, :]

        return deltaH, deltaO

    def updateWeights(self, deltaH, deltaO, data, h, lr, a):

        self.theta = a*self.theta - (1-a) * np.matmul(deltaH, data.T)
        dW = lr*self.theta
        self.W += dW

        self.psi = a*self.psi - (1-a) * np.matmul(deltaO, h.T)
        dV = lr*self.psi
        self.V += dV
        return

    def plot_boundary(self, data, classA, classB, msg, currentD=None):

        ## setting the plot ranges
        xmax = np.max(data[0, :])
        xmax += np.sign(xmax) * 0.2 * xmax

        xmin = np.min(data[0, :])
        xmin -= np.sign(xmin) * 0.2 * xmin

        ymax = np.max(data[1, :])
        ymax += np.sign(ymax) * 0.2 * ymax

        ymin = np.min(data[1, :])
        ymin -= np.sign(ymin) * 0.2 * ymin

        # ## plot normal vector
        # Wn = self.W / np.linalg.norm(self.W)
        # WnOrigin = np.zeros(Wn.shape)


        plt.figure()

        ### computing boundary
        # if abs(self.W[0]/self.W[1]) <= 1:
        #     line_max = (- self.W[0] * xmax - self.W[2]) / self.W[1]
        #     line_min = (- self.W[0] * xmin - self.W[2]) / self.W[1]
        #     WnOrigin[0] = (xmin+xmax)/2
        #     WnOrigin[1] = (line_min+line_max)/2
        #     plt.plot([xmin, xmax], [line_min, line_max ])
        #     print()
        #
        # else:
        #     line_max = (- self.W[1] * ymin - self.W[2]) / self.W[0]
        #     line_min = (- self.W[1] * ymax - self.W[2]) / self.W[0]
        #     WnOrigin[0] = (line_min+line_max)/2
        #     WnOrigin[1] = (ymin+ymax)/2
        #     plt.plot([line_min, line_max], [ymax,ymin ])

        plt.scatter(classA[0, :], classA[1, :], color='red', label="class A")
        plt.scatter(classB[0, :], classB[1, :], color='blue', label="class B")

        # plt.plot([WnOrigin[0], WnOrigin[0] + Wn[0]], [WnOrigin[1], WnOrigin[1] + Wn[1]])
        # plt.plot([0, Wn[0]], [0, Wn[1]])
        # print("Wn:", Wn)


        if currentD is not None:
            plt.plot(currentD[0], currentD[1], "vy")

        plt.legend()
        plt.axis('equal')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title(msg)
        plt.show()
        print("")

    def train_epoch(self, idxTrain, idxVal, targets, data, classA, classB, epochs=100, lr=0.9, a=0):

        for epoch in range(epochs):

            self.train(data[:, idxTrain], targets[idxTrain], lr, a)
            self.validation(data[:, idxVal], targets[idxVal])

        self.vizualize_nn(idxTrain, idxVal, targets, data)

        # self.plot_boundary(data, classA, classB, "final")
        print("")

    def train(self, data, targets, lr, a):

        h, output = self.forward(data)
        deltaH, deltaO = self.backward(h, output, targets)
        self.updateWeights(deltaH, deltaO, data, h, lr, a)

        ## computing average mean square loss and accuracy
        self.loss.append(np.dot(output[0]-targets, output[0]-targets)/targets.shape[0])
        self.acc.append(np.sum(np.abs(output[0]-targets)<1)/targets.shape[0])

    def validation(self, data, targets):

        h, output = self.forward(data)

        ## computing average mean square loss and accuracy
        self.val_loss.append(np.dot(output[0]-targets, output[0]-targets)/targets.shape[0])
        self.val_acc.append(np.sum(np.abs(output[0]-targets)<1)/targets.shape[0])

    def vizualize_nn(self, idxTrain, idxVal, targets, data):

        # ### plotting loss
        # plt.figure()
        # plt.plot(self.loss, label='training loss')
        # plt.plot(self.val_loss, label='validation loss')
        #
        # plt.xlabel('epochs')
        # plt.ylabel('loss')
        # plt.title('Training vs Validation loss')
        # plt.legend()
        # plt.show()
        #
        # ### plotting loss
        # plt.figure()
        # plt.plot(self.acc, label='training accuracy')
        # plt.plot(self.val_acc, label='validation accuracy')
        #
        # plt.xlabel('epochs')
        # plt.ylabel('loss')
        # plt.title('Training vs Validation accuracy')
        # plt.legend()
        # plt.show()
        # print("")


        ### plotting samples
        plt.figure()
        idc = np.where(targets[idxTrain] == -1)[0]
        plt.scatter(data[0,idxTrain][idc], data[1,idxTrain][idc], marker='o', color='red', label="Training Negative Class")
        idc = np.where(targets[idxVal] == -1)[0]
        plt.scatter(data[0, idxVal][idc], data[1, idxVal][idc], marker='x', color='red', label="Validation Negative Class")


        idc = np.where(targets[idxTrain] == 1)[0]
        plt.scatter(data[0,idxTrain][idc], data[1,idxTrain][idc], marker='o', color='blue', label="Training Positive Class")
        idc = np.where(targets[idxVal] == 1)[0]
        plt.scatter(data[0, idxVal][idc], data[1, idxVal][idc], marker='x', color='blue', label="Validation Positive Class")

        #### mesh grid
        xmin = np.min(data[0, :])
        xmax = np.max(data[0, :])
        xR = (xmax-xmin)/100

        ymin = np.min(data[1, :])
        ymax = np.max(data[1, :])
        yR = (ymax-ymin)/100

        x = np.arange(xmin, xmax, xR)
        x = np.repeat(x, 100)

        y = np.arange(ymin, ymax, yR)
        y = np.tile(y, 100)
        b = np.ones(y.shape[0])



        dataBoundary = np.stack((x, y, b), axis=0)

        _, predictions = self.forward(dataBoundary)

        ### plot negative predictions
        idc = np.where(predictions[0] < 0)[0]
        plt.scatter(dataBoundary[0, idc], dataBoundary[1, idc], alpha=0.1, marker='s', color='red', label="Negative Area")
        idc = np.where(predictions[0] > 0)[0]
        plt.scatter(dataBoundary[0, idc], dataBoundary[1, idc],  alpha=0.1,marker='s', color='blue', label="Positive Area")


        plt.legend()
        plt.show()
        print("")

class two_layer_Perceptron_function:


    def __init__(self, inputNodes=3, outputNodes=1, hiddenNodes=15):
        ## generate data
        self.W = np.random.random([hiddenNodes-1, inputNodes])
        self.V = np.random.random([outputNodes, hiddenNodes])
        self.theta = 0
        self.psi = 0
        self.loss =[]
        self.val_loss =[]

    def activation(self, input):
        return 1/(1+np.exp(-input))-0.5

    def activation_derivative(self, input):
        return (1+input)*(1-input)

    def forward(self, input):
        hin = np.matmul(self.W, input)
        h = self.activation(hin)

        ## adding ones
        hout = np.ones([h.shape[0]+1, h.shape[1]])
        hout[:-1, :] = h

        oin = np.matmul(self.V, hout)
        output = self.activation(oin)
        return hout, output

    def backward(self, h, output, targets):

        deltaO = (output-targets) * self.activation_derivative(output)
        deltaH = np.matmul(self.V.T, deltaO) * self.activation_derivative(h)
        deltaH = deltaH[:-1, :]

        return deltaH, deltaO

    def updateWeights(self, deltaH, deltaO, data, h, lr, a):

        self.theta = a*self.theta - (1-a) * np.matmul(deltaH, data.T)
        dW = lr*self.theta
        self.W += dW

        self.psi = a*self.psi - (1-a) * np.matmul(deltaO, h.T)
        dV = lr*self.psi
        self.V += dV
        return

    def plot_boundary(self, data, classA, classB, msg, currentD=None):

        ## setting the plot ranges
        xmax = np.max(data[0, :])
        xmax += np.sign(xmax) * 0.2 * xmax

        xmin = np.min(data[0, :])
        xmin -= np.sign(xmin) * 0.2 * xmin

        ymax = np.max(data[1, :])
        ymax += np.sign(ymax) * 0.2 * ymax

        ymin = np.min(data[1, :])
        ymin -= np.sign(ymin) * 0.2 * ymin

        # ## plot normal vector
        # Wn = self.W / np.linalg.norm(self.W)
        # WnOrigin = np.zeros(Wn.shape)


        plt.figure()

        ### computing boundary
        # if abs(self.W[0]/self.W[1]) <= 1:
        #     line_max = (- self.W[0] * xmax - self.W[2]) / self.W[1]
        #     line_min = (- self.W[0] * xmin - self.W[2]) / self.W[1]
        #     WnOrigin[0] = (xmin+xmax)/2
        #     WnOrigin[1] = (line_min+line_max)/2
        #     plt.plot([xmin, xmax], [line_min, line_max ])
        #     print()
        #
        # else:
        #     line_max = (- self.W[1] * ymin - self.W[2]) / self.W[0]
        #     line_min = (- self.W[1] * ymax - self.W[2]) / self.W[0]
        #     WnOrigin[0] = (line_min+line_max)/2
        #     WnOrigin[1] = (ymin+ymax)/2
        #     plt.plot([line_min, line_max], [ymax,ymin ])

        plt.scatter(classA[0, :], classA[1, :], color='red', label="class A")
        plt.scatter(classB[0, :], classB[1, :], color='blue', label="class B")

        # plt.plot([WnOrigin[0], WnOrigin[0] + Wn[0]], [WnOrigin[1], WnOrigin[1] + Wn[1]])
        # plt.plot([0, Wn[0]], [0, Wn[1]])
        # print("Wn:", Wn)


        if currentD is not None:
            plt.plot(currentD[0], currentD[1], "vy")

        plt.legend()
        plt.axis('equal')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title(msg)
        plt.show()
        print("")

    def train_epoch(self, idxTrain, idxVal, targets, data, epochs=100, lr=0.9, a=0):

        for epoch in range(epochs):

            self.train(data[:, idxTrain], targets[idxTrain], lr, a)
            self.validation(data[:, idxVal], targets[idxVal])

        self.vizualize_nn(idxTrain, idxVal, targets, data)

        # self.plot_boundary(data, classA, classB, "final")
        print("")

    def train(self, data, targets, lr, a):

        h, output = self.forward(data)
        deltaH, deltaO = self.backward(h, output, targets)
        self.updateWeights(deltaH, deltaO, data, h, lr, a)

        ## computing average mean square loss and accuracy
        self.loss.append(np.dot(output[0]-targets, output[0]-targets)/targets.shape[0])

    def validation(self, data, targets):

        h, output = self.forward(data)

        ## computing average mean square loss and accuracy
        self.val_loss.append(np.dot(output[0]-targets, output[0]-targets)/targets.shape[0])

    def vizualize_nn(self, idxTrain, idxVal, targets, data):

        # ### plotting loss
        plt.figure()
        plt.plot(self.loss, label='training loss')
        plt.plot(self.val_loss, label='validation loss')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Training vs Validation loss')
        plt.legend()
        plt.show()

        ### plotting ground truth
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.scatter(data[0, :], data[1, :], targets, alpha=0.1 ,marker='o', color='red', label="Ground Truth")
        ### plotting training samples
        _, predictions = self.forward(data[:, idxTrain])

        ax.scatter(data[0, idxTrain], data[1, idxTrain], predictions[0], marker='+', color='blue', label="Training Samples")
        ### plotting validation samples
        _, predictions = self.forward(data[:, idxVal])
        ax.scatter(data[0, idxVal], data[1, idxVal], predictions[0], marker='x', color='green', label="Validation Samples")

        plt.legend()
        plt.show()

        return

class three_layer_network(nn.Module):

    def __init__(self, nh1=3, nh2=2):
        super(three_layer_network, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(5, nh1)
        self.fc2 = nn.Linear(nh1, nh2)
        self.fc3 = nn.Linear(nh2, 1)
        self.dropout = nn.Dropout(0.1)
        self.train_loss = []
        self.val_loss = []
        self.test_loss =[]
        self.lowest_val = np.inf
        self.patience = 0

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        x = self.sigmoid(x)
        # x = self.dropout(x)

        x = self.fc3(x)
        return x

    def train_epoch(self, optimizer, data_train, target_train, data_val, target_val, data_test, target_test, epochs=1000):

        for epoch in range(epochs):

            self.train()
            ## train
            outputTrain = self.forward(torch.from_numpy(data_train).type(torch.float))

            loss = torch.sum((torch.from_numpy(target_train).type(torch.float) - outputTrain[:, 0])**2 )/ outputTrain.shape[0]
            loss.backward()

            ## updating NN-weights
            optimizer.step()
            ## resetting optimizer
            optimizer.zero_grad()
            self.train_loss.append(loss.data.cpu().numpy())

            self.eval()

            ## val
            outputVal = self.forward(torch.from_numpy(data_val).type(torch.float))
            loss = torch.sum((torch.from_numpy(target_val).type(torch.float) - outputVal[:, 0])**2 )/ outputVal.shape[0]
            self.val_loss.append(loss.data.cpu().numpy())

            ## test
            outputTest = self.forward(torch.from_numpy(data_test).type(torch.float))
            loss = torch.sum((torch.from_numpy(target_test).type(torch.float) - outputTest[:, 0])**2 )/ outputTest.shape[0]
            self.test_loss.append(loss.data.cpu().numpy())

            if self.val_loss[-1] < 0.95 * self.lowest_val:
                self.patience = 0
                self.lowest_val = self.val_loss[-1]
                torch.save(self.state_dict(), "best_model.pth")
                print("Best model found at", epoch)
            else:
                self.patience += 1

            if self.patience == 200:
                break


        self.load_state_dict(torch.load("best_model.pth"))
        outputTrain = self.forward(torch.from_numpy(data_train).type(torch.float))
        outputVal = self.forward(torch.from_numpy(data_val).type(torch.float))
        outputTest = self.forward(torch.from_numpy(data_test).type(torch.float))

        # ### plotting forecasting
        plt.figure()
        plt.plot(target_train, label='training groundtruth')
        plt.plot(outputTrain.data.cpu().numpy(), label='training forecasting')
        plt.legend()
        plt.show()

        # ### plotting loss
        plt.figure()
        plt.plot(self.train_loss, label='training loss')
        plt.plot(self.val_loss, label='validation loss')
        plt.plot(self.test_loss, label='test loss')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Training/Validation/Test loss')
        plt.legend()
        plt.show()
        print("")


