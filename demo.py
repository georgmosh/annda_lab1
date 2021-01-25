import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

def generateDatapoints():
    np.random.seed(0)
    
    num_of_datapoints = 100

    means_A = np.zeros(2)
    means_A[0] += 0.5
    means_A[1] += 1

    sigma_A = 0.5

    means_B = np.zeros(2)
    means_B -= 1

    sigma_B = 0.5

    class_A1 = np.random.normal(loc=0, scale=1, size=num_of_datapoints) * sigma_A + means_A[0]
    class_A2 = np.random.normal(loc=0, scale=1, size=num_of_datapoints) * sigma_A + means_A[1]
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

def main():
    num_epochs = 100000
    num_classes = 2
    lr = 0.3
    num_of_datapoints, class_A, class_B = generateDatapoints()
    x_axis = [class_A[0], class_B[0]]
    y_axis = [class_A[1], class_B[1]]
    
    class_A_all = [class_A[0], class_A[1], np.zeros(num_of_datapoints)]
    class_B_all = [class_B[0], class_B[1], np.ones(num_of_datapoints)]
    
    data = np.transpose(np.concatenate((class_A_all, class_B_all), axis=1))
    np.random.shuffle(data)
    
    W = np.random.rand(3) #first weight is the bias
    point_1 = [0, (-W[0]/W[2])]
    point_2 = [(-W[0]/W[1]), 0]
    plot_with_boundary(class_A[0], class_A[1], class_B[0], class_B[1], point_1, point_2)
    
    print("INITLAL", W)
    for epoch in range(0, num_epochs):
        for i in range(num_classes * num_of_datapoints):
            x = [1, data[i][0], data[i][1]]
            x = np.array(x)
            t = data[i][2]
            prediction = np.dot(np.transpose(W), x)
            if(prediction > 0):
                if(t == 0):
                    delta = -x
                    W = W + lr * delta
            else:
                if(t == 1):
                    delta = x
                    W = W + lr * delta
    
    point_1 = [0, (-W[0]/W[2])]
    point_2 = [(-W[0]/W[1]), 0]
    plot_with_boundary(class_A[0], class_A[1], class_B[0], class_B[1], point_1, point_2)
    print(W)

main()