import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import torch
import torch.optim as optim

## setting seed
set_set(0)


data_train, target_train, data_val, target_val, data_test, target_test = generate_time_series(N=1500, beta=0.2, gamma=0.1, n=10, tau=25)


model = three_layer_network(nh1=3, nh2=5)


optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)#, weight_decay=0.001)
model.train_epoch( optimizer, data_train, target_train, data_val, target_val, data_test, target_test, epochs=10000)

print("")