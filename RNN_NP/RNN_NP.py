import numpy as np
import time

def relu(x, d=0):
    #d==True during backprop
    x[x < 0] = 0
    if d: x = np.where(x <= 0, 1, 0)
    return x

def softmax(x, d=0):
    #d==True during backprop
    x = np.exp(x) / sum(np.exp(x))
    if d: x = np.where(x < 0, 0, 1)
    return x

def KL_divergence(X, Y, d=0):
    #d==True during backprop
    #Derived formula obtained from http://proceedings.mlr.press/v37/theis15-supp.pdf
    loss = - Y.dot(np.log(X)) if not d else np.dot((X-Y).T, np.identity(X.shape[0]))
    return loss

# Shape(3, 3)
# 3 input neurons and one layer with 10 hidden neurons and an output of three neurons

def rnn(shape, activation):
    # [3, 10, 3 ,1]
    # return layer_stack
    pass

def initialize_layer(previous_layer_neurons, this_layer_neurons, 
                     activation_function=relu): 
    size = (this_layer_neurons, previous_layer_neurons)
    weights = np.random.standard_normal(size)
    # http://cs231n.github.io/neural-networks-2/ initialized at 0.001
    biases = np.full(previous_layer_neurons, 0.001)
    previous_activation = np.zeros(previous_layer_neurons)
    previous_pre_activation = np.zeros(previous_layer_neurons)
    layer = dict([("W", weights), ("b", biases), ("f", activation_function), 
                  ("a", previous_activation)])
    return layer

def activate_layer(layer, d=0):
    #this_layer_weights
    W = layer["W"]
    #previous_layer_output
    a = layer["a"]
    #bias
    b = layer["b"]
    #activation_function
    f = layer["f"]
    #logits aka. pre-nonlinearity activation
    z = W.dot(a) + b
    return f(z, d=d)

def add_layer(layer, L=[]):
    #a layer must be a dict of weigts, bias, activation function, 
    #and a tensor of the previous layers activation output
    return L + [layer]

def feed_input_to_stack(input_data, layer_stack):
    #set previous_layer_activation of first layer to be the input
    layer_stack[0]["a"] = input_data
    return layer_stack

def forward_pass(layer_stack):
    #previous_activation output
    out = None
    for i, layer in enumerate(layer_stack):
        if out is not None: layer_stack[i]["a"] = out
        out = activate_layer(layer)
    return out, layer_stack

def back_prop(layer_stack, dloss):
    for layer in reversed(layer_stack[:-1]):
        #I recompute z to save some memory and lines of code
        out = activate_layer(layer, d=True)
        #previous_activation output
        a = layer["a"]
        x = np.multiply(dloss, out)
        print(x)
        dz = x.dot(a)
        break

input_data = np.array([0 for _ in range(9)] + [1])    
    
#layer_stack = add_layer(initialize_layer(100, 3, relu))
#layer_stack = add_layer(initialize_layer(3, 10, relu), layer_stack)
#layer_stack = add_layer(initialize_layer(10, 3, relu), layer_stack)
#layer_stack = add_layer(initialize_layer(3, 100, softmax), layer_stack)
#layer_stack = feed_input_to_stack(input_data, layer_stack)

layer_stack = add_layer(initialize_layer(5, 10, relu))
layer_stack = add_layer(initialize_layer(5, 5, relu))
layer_stack = add_layer(initialize_layer(10, 5, softmax))

out, layer_stack = forward_pass(layer_stack)
loss = KL_divergence(out, input_data)
dloss = KL_divergence(out, input_data, d=True)
print(loss)
print(dloss)
print(out)

back_prop(layer_stack, dloss)
#for l, y in layer_stack: print(l,)
#activate_layer(W1, b1, relu)