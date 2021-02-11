# part 4.3.   Data Clustering: Votes of MPs


import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

################## HELPER FUNCTIONS ##################
def eucledianDistance(a, b):
    # return  np.sqrt(np.sum(np.square(center - CLunit)) ) # equivalent
    return np.linalg.norm(a - b)

def manhattanDistance(a, b):
    return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

def updateRule(x, w, learning_rate):
    return learning_rate * (x - w)

def computeNeighbourhood(iteration, num_iteration, sigma_zero):
    lambda_t = num_iteration / np.log(sigma_zero)
    sigma_t = sigma_zero*np.exp(-(float(iteration)/lambda_t))
    return sigma_t

def toGrid(index, gridSide):
    return (index % gridSide, index // gridSide)

def fromGrid(i, j, gridSide):
    return i + j * gridSide

def readFile(filename):
    with open(filename) as f:
        content = f.read()
    dataset = content.split(",")
    training_data = np.array(dataset)
    training_data = training_data.reshape((349, 31))
    return training_data

def plot(X, y, sizes, X1, y1, sizes1):
    plt.scatter(X, y, color='b', s=sizes, marker='+')
    plt.scatter(X1, y1, color='r', s=sizes1, marker='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def alreadyInList(x, y, Lx, Ly):
    already = False
    coord = -1
    for i in range(0, len(Lx)):
        if(Lx[i] == x):
            if(Ly[i] == y):
                already = True
                coord = i

    return already, coord

def main():
    print("main")

    #hyperparameters
    epochs = 20
    num_nodes = 100 # number of output layer nodes
    learning_rate = 0.2
    initial_neighbourhood = 50
    latend_dimension = 10

    # read data from file - build matrix props 32x84
    props = readFile("votes.dat")
    props = props.astype(float)
    num_MPs = len(props)
    
    # initialise weight matrix 349 con pesi uniformi 0-1
    weights = np.random.rand(num_nodes, props.shape[1])

    # SOM training
    for e in range(0, epochs):
        for id in range (0, num_MPs): # id è MP ID
            votes = props[id]
            # find winning node "closest to the selected MP's votes"
            distances = np.zeros(num_nodes)
            for j in range(0, num_nodes):
                distances[j] = eucledianDistance(votes, weights[j])
            arg_sorted_dist = np.argsort(distances)
            winning_node = arg_sorted_dist[0]
            coord2D_winner = toGrid(winning_node, latend_dimension)
            # update w
            neighbourhood = computeNeighbourhood(e, epochs, initial_neighbourhood)
            distances = {}
            for i in range(0, num_nodes):
                coord2D = toGrid(i, latend_dimension)
                bmu_dist = manhattanDistance(coord2D, coord2D_winner)
                distances[i] = bmu_dist
            distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
            for item in distances.items():
                if(item[0] != winning_node):
                    tn = np.exp(-((item[1]**2)/(2*(neighbourhood**2))))
                    weights[item[0]] += tn * updateRule(votes, weights[item[0]], learning_rate)
    
    # print MPs in the clustered order
    pos = np.zeros(num_MPs)
    for MP in range(0, num_MPs):  # id è MP ID
        votes = props[MP]
        # find winning node "closest to the selected MP's votes"
        distances = np.zeros(num_nodes)
        for node in range(0, num_nodes):
            distances[node] = eucledianDistance(votes, weights[node])
        arg_sorted_dist = np.argsort(distances)
        winning_node = arg_sorted_dist[0]
        pos[MP] = winning_node

    # read MPs' sex data
    file1 = open("mpsex.dat", 'r')
    Lines = file1.readlines()

    sexData = []
    # Strips the newline character
    for line in Lines:
        sexData.append(int(line))
    sexData = np.array(sexData)
    
    # convert indices to points in a 2-D topological map
    x_axis_blue = []
    y_axis_blue = []
    x_axis_red = []
    y_axis_red = []
    point_size_blue = []
    point_size_red = []

    for i in range(0, num_MPs):
        win1D = pos[i]
        color = sexData[i]
        win2D = toGrid(win1D, latend_dimension)
        if(color == 0):
            data_frame_info = alreadyInList(win2D[0], win2D[1], x_axis_blue, y_axis_blue)
            if(data_frame_info[0] == False):
                x_axis_blue.append(win2D[0])
                y_axis_blue.append(win2D[1])
                point_size_blue.append(10)
            else:
                point_size_blue[data_frame_info[1]] += 10
        else:
            data_frame_info = alreadyInList(win2D[0], win2D[1], x_axis_red, y_axis_red)
            if(data_frame_info[0] == False):
                x_axis_red.append(win2D[0])
                y_axis_red.append(win2D[1])
                point_size_red.append(10)
            else:
                point_size_red[data_frame_info[1]] += 10

    plot(x_axis_blue, y_axis_blue, point_size_blue, x_axis_red, y_axis_red, point_size_red)

if __name__ == "__main__":
    main()