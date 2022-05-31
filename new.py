import numpy as np
from math import exp
import pandas as pd
import random
import matplotlib.pyplot as plt
import sys

def sigmoid(x):
  return 1 / (1 + exp(-x))

class NN():

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.hidden_layer = np.random.rand(5,4+1)*2 - 1 # matrix
        self.output_layer = np.random.rand(1,5+1)*2 - 1 # matrix
        self.layers = [self.hidden_layer, self.output_layer]
        self.deltas = []

    def forward(self, inp):
        assert type(inp) is np.ndarray
        assert 1 == inp.ndim
        self.outputs = [None]*len(self.layers)

        for layer_index in range(len(self.layers)):
            previous_input = np.append(inp, 1) if layer_index == 0 else np.append(self.outputs[layer_index - 1], 1)
            layer_output = np.array([])
            for nueron in self.layers[layer_index]:
                layer_output = np.append(layer_output, sigmoid(nueron.dot(previous_input)))

            self.outputs[layer_index] = layer_output

        return self.outputs[-1]

    def backprop(self, inp, expected):
        assert type(expected) is np.ndarray and type(inp) is np.ndarray
        assert 1 == expected.ndim and 1 == inp.ndim
        self.deltas = [None]*len(self.layers)

        for layer_index in reversed(range(len(self.layers))):
            delta_output = np.array([])
            for nueron_index in range(len(self.layers[layer_index])):
                output = self.outputs[layer_index][nueron_index]

                if layer_index == len(self.layers) - 1:
                    delta = (output - expected[nueron_index]) * output * (1 - output)
                else:
                    error = self.layers[layer_index + 1][:, nueron_index].dot(self.deltas[layer_index + 1])
                    delta = output * (1 - output) * error
                    
                delta_output = np.append(delta_output, delta)
            
            self.deltas[layer_index] = delta_output

        self.__update(inp)

    def __update(self, inp):
        for layer_index in range(len(self.layers)):
            previous_input = np.append(inp, 1) if layer_index == 0 else np.append(self.outputs[layer_index - 1], 1)
            for nueron_index in range(len(self.layers[layer_index])):
                self.layers[layer_index][nueron_index] -= self.deltas[layer_index][nueron_index] * self.learning_rate * previous_input

        

model = NN(learning_rate=0.01)

"""
Even num 1a then output should be 0, else 1
"""

dataset = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1),
       (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)]

epochs = 1000

errors = []

for e in range(epochs):
    print("Epoch {}/{}".format(e + 1, epochs))
    random.shuffle(dataset)

    error = 0
    
    for inp in dataset:
        output = model.forward(np.asarray(inp))[0]
        output = 1 if output >= 0.5 else 0
        model.backprop(np.asarray(inp), np.array([inp.count(1) % 2 == 0]).astype(int))

        if output != inp.count(1) % 2:
          error += 1
        
    
    errors.append(error/16.0)


plt.plot(errors)
plt.ylim(0,1)
plt.savefig("./myfig.png")
plt.close()





