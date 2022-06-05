import numpy as np
from random import randint
import matplotlib.pyplot as plt

class NN():

    def __init__(self, hidden_layers=0, layer_depth_array=[1], learning_rate=0.01):
        assert hidden_layers == len(layer_depth_array) - 1
        assert hidden_layers >= 0
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

        if type(self.layers) is tuple: # Layers needs to be initialized with weights to input layer
            layers = self.layers[0]
            input_shape = (layers[0].shape[0], (len(inp) if inp.ndim == 1 else len(inp[0])) + 1)
            self.layers = [np.random.rand(*input_shape)*2 - 1, *layers[1:]]
        self.outputs = [None]*len(self.layers)

        for layer_index in range(len(self.layers)):
            previous_input = inp if layer_index == 0 else self.outputs[layer_index - 1]
            self.outputs[layer_index] = np.array([])
            for nueron in self.layers[layer_index]:
                self.outputs[layer_index] = np.append(self.outputs[layer_index], self.sigmoid(np.dot(nueron, np.append(previous_input, 1))))

        return self.outputs[-1]

    def backprop(self, inp, expected):
        assert type(expected) is np.ndarray and type(inp) is np.ndarray
        assert inp.ndim == expected.ndim

        self.deltas = [None]*len(self.layers)
        cost_total = expected - self.outputs[-1] ##np.power(expected - self.outputs[-1], 2).sum()*0.5
        self.deltas[-1] = cost_total * self.sigmoid_prime_y(self.outputs[-1])

        # for hidden layers
        for layer_index in reversed(range(len(self.layers) - 1)):
            delta_list = self.layers[layer_index + 1][:,:-1].T @ self.deltas[layer_index + 1]
            self.deltas[layer_index] = self.sigmoid_prime_y(self.outputs[layer_index]) * delta_list

        # update weights
        for layer_index in range(len(self.layers)):
            previous_input = inp if layer_index == 0 else self.outputs[layer_index - 1]
            for nueron_index in range(len(self.layers[layer_index])):
                self.layers[layer_index][nueron_index] += self.deltas[layer_index][nueron_index] * self.learning_rate * np.append(previous_input, 1)

        return np.average(np.abs(expected - self.outputs[-1]))

    #def train(self, inp, output, epochs=10000):
        for e in range(epochs):
            self.forward(inp)
            final_err = self.backprop(inp, output)
        return final_err


model = NN(learning_rate=0.1, layer_depth_array=[1], hidden_layers=0)

# Data for input and network object idea from Aidan Wilson @ https://towardsdatascience.com/inroduction-to-neural-networks-in-python-7e0b422e6c24
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

# Iterate for n epochs
for e in range(epochs):
    print("Epoch {}/{}".format(e + 1, epochs), end='\r')

    error = 0
    selected_index = randint(0, len(inputs) - 1)
    inp = inputs[selected_index]
    expected_output = outputs[selected_index]

    model.forward(inp)
    cost_total = model.backprop(inp, expected_output) # summed avg err returned
    #print(inp, cost_total)
    #print("Given input {}, I output {}, expecting {}, yielding error {}".format(inp, model.outputs[-1], expected_output, cost_total))
    errors.append(cost_total)

print(model.layers)
    
plt.plot([sum(errors[i:i+6])/6.0 for i in range(0,len(errors),6)])
plt.title('Summed Avg Error for Nueral Network')
plt.xlabel('epoch / 6')
plt.ylabel('sum avg error (expected & output layer)')
plt.ylim(0,1)
plt.savefig("./myfig.png")
plt.close()
