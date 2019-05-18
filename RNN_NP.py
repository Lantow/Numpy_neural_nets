import numpy as np

def relu(x):
    x[x < 0] = 0   
    return x

# Shape(3, 3)
# 3 input neurons and one layer with 10 hidden neurons and an output of three neurons

def rnn(shape, activation):
    # [3, 10, 3 ,1]
    # return layer_stack
    pass

def initialize_layer(this_layer_neurons, previous_layer_neurons, 
                     activation=relu): 
    size = (this_layer_neurons, previous_layer_neurons)
    weights = np.random.standard_normal(size)
    # http://cs231n.github.io/neural-networks-2/ initialized at 0.001
    biases = np.full(previous_layer_neurons, 0.001)
    return weights, biases, activation

def activate_layer(previous_layer_weights, this_layer_weiths, 
                   bias, activation_function):
    z = np.dot(this_layer_weiths, previous_layer_weights)+ bias
    return activation_function(z)

def add_layer(layer, L=[]):
    return L + [layer]

def forward_pass(layer_stack, input_data):
    weights0 = input_data
    while layer_stack:
        weights1, bias, activation = layer_stack.pop(0)
        weights0 = activate_layer(weights1, weights0, 
                                  bias, activation)
        
    return weights0

input_data = np.array([0 for _ in range(99)] + [1])

layer_stack = add_layer(initialize_layer(100, 3, relu))
layer_stack = add_layer(initialize_layer(3, 10, relu), layer_stack)
layer_stack = add_layer(initialize_layer(10, 3, relu), layer_stack)
layer_stack = add_layer(initialize_layer(3, 100, relu), layer_stack)

forward_pass(layer_stack, input_data)

#for l, y in layer_stack: print(l,)
#activate_layer(W1, b1, relu)