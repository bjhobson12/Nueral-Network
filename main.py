import numpy as np
from math import exp
import pandas as pd
import random
import matplotlib.pyplot as plt
import sys

class NN():

    def __init__(self, hidden_layers=1, layer_depth_array=[5, 1], learning_rate=0.01):
        assert hidden_layers == len(layer_depth_array) - 1
        assert hidden_layers >= 1
        self.learning_rate = learning_rate
        self.layers = ([np.random.rand(layer_depth_array[i], layer_depth_array[i-1] + 1 if i > 0 else 1)*2 - 1 for i in range(len(layer_depth_array))],)
        self.deltas = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime_y(self, y):
        return y * (1 - y) 

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, inp):
        assert type(inp) is np.ndarray
        assert 1 == inp.ndim
        if type(self.layers) is tuple: # Layers needs to be initialized with weights to input layer
            layers = self.layers[0]
            input_shape = (layers[0].shape[0], inp.size + 1)
            self.layers = [np.random.rand(*input_shape)*2 - 1, *layers[1:]]
        self.outputs = [None]*len(self.layers)

        for layer_index in range(len(self.layers)):
            previous_input = np.append(inp, 1) if layer_index == 0 else np.append(self.outputs[layer_index - 1], 1)
            self.outputs[layer_index] = np.array([])
            for nueron in self.layers[layer_index]:
                o = self.sigmoid(nueron.dot(previous_input))
                self.outputs[layer_index] = np.append(self.outputs[layer_index], o)

        return self.outputs[-1]

    def backprop(self, inp, expected):
        assert type(expected) is np.ndarray and type(inp) is np.ndarray
        assert 1 == expected.ndim and 1 == inp.ndim
        self.deltas = [None]*len(self.layers)

        cost_total = np.power(expected - self.outputs[-1], 2).sum()*0.5

        self.deltas[-1] = cost_total * self.outputs[-1] * (1 - self.outputs[-1])  # derivative of sigmoid

        for layer_index in reversed(range(len(self.layers) - 1)):
            delta_list = self.layers[layer_index + 1][:,:-1].T @ self.deltas[layer_index + 1]
            self.deltas[layer_index] = self.outputs[layer_index] * (1 - self.outputs[layer_index]) * delta_list

        self.__update(inp)

        return np.average(np.abs(expected - self.outputs[-1]))

    def __update(self, inp):
        for layer_index in range(len(self.layers)):
            previous_input = np.append(inp, 1) if layer_index == 0 else np.append(self.outputs[layer_index - 1], 1)
            for nueron_index in range(len(self.layers[layer_index])):
                self.layers[layer_index][nueron_index] -= self.deltas[layer_index][nueron_index] * self.learning_rate * previous_input



model = NN(learning_rate=0.1, layer_depth_array=[2, 1], hidden_layers=1)

"""
Even num 1a then output should be 0, else 1
"""

dataset = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1),
       (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)]

# input data
inputs = np.array([[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 0, 1]])
# output data
outputs = np.array([[0], [0], [0], [1], [1], [1]])

epochs = 10000

errors = []

for e in range(epochs):
    print("Epoch {}/{}".format(e + 1, epochs))

    error = 0

    random.shuffle(inputs)
    
    inp = inputs[0]
    expected_output = outputs[0]
    output = model.forward(inp)[0]
    #print(output, 0 if output <= 0.5 else 1)
    output = 0 if output <= 0.5 else 1
    # output = 1 if output >= 0.5 else 0
    cost_total = model.backprop(inp, expected_output)
        
    
    errors.append(cost_total)
    

plt.plot(errors)
plt.ylim(0,1)
plt.savefig("./myfig.png")
plt.close()











