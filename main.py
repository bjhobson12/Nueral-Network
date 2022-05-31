from random import uniform as rand
from math import exp, copysign
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

class NN():

    def __init__(self, hidden_layers=1, layer_depth_array=[5, 1], learning_rate=0.01):
        assert hidden_layers == len(layer_depth_array) - 1
        assert hidden_layers >= 1
        self.learning_rate = learning_rate
        # add one for final layer
        self.layers = [[] for i in range(hidden_layers + 1)]
        self.deltas = [[] for i in range(hidden_layers + 1)] # add one for final layer
        self.outputs = [[None]*layer_depth_array[i] for i in range(hidden_layers + 1)]  # add one for final layer

        # initialize arrays for each node in layer to store list of weights
        for i in range(0, len(self.layers)):
            # bias added manually
            self.layers[i].extend([[] for j in range(layer_depth_array[i])])
            self.deltas[i].extend([None]*layer_depth_array[i])

    def activation(self, x):
        if abs(x) > 709:
            return 1.0/(1.0 + exp(copysign(709, x)))
        return 1.0/(1.0 + exp(-x))

    def activation_derivitive(self, x):
        return self.activation(x) * (1 - self.activation(x))

    def forward(self, inp):
        # Iterates from left to right over layers of network
        for layer_index in range(len(self.layers)):
            # Iterates from top to bottom of a layer
            previous = lambda mat: mat[layer_index - 1] if layer_index > 0 else inp
            for depth in range(len(self.layers[layer_index])):
                # initialize weights on very first forward propagate to random values
                if self.layers[layer_index][depth] == []:
                    self.layers[layer_index][depth] = [rand(-1, 1) for i in range(1 + len(previous(self.layers)))]  # add one for bias

                # Initialize sum to bias weight
                sigma = self.layers[layer_index][depth][-1]
                # since zip function only iterates over smallest list, the bias in the layers[index][depth] list will be ignored
                for weight, local_input in zip(self.layers[layer_index][depth], previous(self.outputs)):
                    sigma += weight * local_input

                self.outputs[layer_index][depth] = self.activation(sigma)

        return self.outputs[layer_index]

    def backprop(self, inp, expected):
        assert len(expected) == len(self.layers[-1])
        # Iterates from right to left over layers of network
        for layer_index in reversed(range(len(self.layers))):
            # Iterates from top to bottom of a layer

            for depth in range(len(self.layers[layer_index])):
                output = self.outputs[layer_index][depth]
                
                if layer_index == len(self.layers) - 1:  # Output layer
                    # James Collins stamp of approval (soa)
                    delta = (output - expected[depth]) * output * (1 - output) # Derivative of sigmoid
                    # End James Collins stamp of approval (esoa)

                    # Begin debugging more here
                else:
                    error = 0
                    for nuerons_in_next_layer in range(len(self.layers[layer_index + 1])):
                        w = self.layers[layer_index + 1][nuerons_in_next_layer][depth]
                        nueron_delta = self.deltas[layer_index + 1][nuerons_in_next_layer]
                        error += w * nueron_delta
                    delta = output * (1 - output) * error #derivative of sigmoid
                    
                self.deltas[layer_index][depth] = delta

        # Iterates from right to left over layers of network
        for layer_index in reversed(range(len(self.layers))):
            # Iterates from top to bottom of a layer
            previous = lambda w: self.outputs[layer_index - 1][w] if layer_index > 0 else inp[w]

            for depth in range(len(self.layers[layer_index])):
                # Iterates from top to bottom of weights to node
                for weight in range(len(self.layers[layer_index][depth]) - 1): # subtract one for bias
                    previous_output = previous(weight)
                    self.layers[layer_index][depth][weight] += -self.learning_rate * self.deltas[layer_index][depth] * previous_output

                self.layers[layer_index][depth][weight] += self.learning_rate * self.deltas[layer_index][depth]

    def __str__(self):
        return str(self.outputs)

model = NN(learning_rate=0.1, layer_depth_array=[15,20,3], hidden_layers=2)

"""
Even num 1a then output should be 0, else 1
"""

dataset = load_wine()

epochs = 100

accs = []

for e in range(epochs):
    print("Epoch {}/{}".format(e+1, epochs), end='\r')

    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target)

    local_acc = 0
    for x, y in zip(list(X_train), list(y_train)):
        output = model.forward(list(x))[0]
        #print(output)
        output = 1 if output >= 0.5 else 0
        if int(y) == output:
            local_acc += 1
        model.backprop(list(x), [int(y)])
    
    accs.append(float(local_acc) / y_train.size)
        
plt.plot(accs)
plt.ylim(0,1)
plt.savefig("./myfig.png")
plt.close()



