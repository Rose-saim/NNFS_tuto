# Init data
x = [1.0, -2.0, 3.0] # input values
w = [-3.0, -1.0, 2.0] # weights
b = 1.0 # bias

# Multiplying inputs by weigths -> Create neurons
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# Compute sum every neurons and a bias
z = xw0 + xw1 + xw2 + b

# ReLU activation function
y = max(z, 0)
print("ReLU| ", y)
# The derivative from the next layer
dvalue = 1.0

# Derivative of ReLU and the chain rule
drelu_dz = dvalue * (1. if z > 0 else 0.)
print(drelu_dz)

# Partial derivatives of the multiplication, the chain rule
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

# Partial derivatives of the multiplication, the chain rule
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]
drelu_dx0 = dvalue * (1. if z > 0 else 0.) * w[0]
drelu_dx1 = dvalue * (1. if z > 0 else 0.) * w[1]
drelu_dx2 = dvalue * (1. if z > 0 else 0.) * w[2]
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dw2 = drelu_dxw2 * dmul_dw2
print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)
dw = [drelu_dw0, drelu_dw1, drelu_dw2]
db = drelu_db
print(w, b)
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db
print(w, b)
print(drelu_dx0)


# Multiplying inputs by weigths -> Create neurons
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# Compute sum every neurons and a bias
z = xw0 + xw1 + xw2 + b

# ReLU activation function
y = max(z, 0)
print("ReLU| ", y)

import numpy as np

# Passes in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[1., 1., 1.]])

# We have 3 set of weights - one set for each neuron
# we have 4 imputs, this 4 weigths
# recall that we keep weigths tranposed
weigths = np.array([[0.2, 0.8, -0.5, 1],
			[0.5, -0.91, 0.26, -0.5],
			[-0.26, -0.27, 0.17, 0.87]]).T

# sum weigths of given input
# and multiply by the passed in gradiant for this neuron
dx0 = sum(weigths[0])*dvalues[0]
dx1 = sum(weigths[1])*dvalues[0]
dx2 = sum(weigths[2])*dvalues[0]
dx3 = sum(weigths[3])*dvalues[0]

dinputs = np.array([dx0, dx1, dx2, dx3])
print(dinputs)

# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[1., 1., 1.]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, of thus 4 weights
# recall that we keep weights transposed
weigths = np.array([[0.2, 0.8, -0.5, 1],
			[0.5, -0.91, 0.26, -0.5],
			[-0.26, -0.27, 0.17, 0.87]]).T

# sum weigths of given input
# and multiply by the passed in gradiant for this neuron
dx0 = sum(weigths[0]*dvalues[0])
dx1 = sum(weigths[1]*dvalues[0])
dx2 = sum(weigths[2]*dvalues[0])
dx3 = sum(weigths[3]*dvalues[0])

dinputs = np.array([dx0, dx1, dx2, dx3])
print(dinputs)

# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[1., 1., 1.]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, of thus 4 weights
# recall that we keep weights transposed
weigths = np.array([[0.2, 0.8, -0.5, 1],
			[0.5, -0.91, 0.26, -0.5],
			[-0.26, -0.27, 0.17, 0.87]]).T

# sum weigths of given input
# and multiply by the passed in gradiant for this neuron
dinputs = np.dot(dvalues[0], weigths.T)
print(dinputs)

# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[1., 1., 1.],
			[2., 2., 2.],
			[3., 3., 3.]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, of thus 4 weights
# recall that we keep weights transposed
weigths = np.array([[0.2, 0.8, -0.5, 1],
			[0.5, -0.91, 0.26, -0.5],
			[-0.26, -0.27, 0.17, 0.87]]).T

# sum weigths of given input
# and multiply by the passed in gradiant for this neuron
dinputs = np.dot(dvalues, weigths.T)
print(dinputs)

# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[1., 1., 1.],
			[2., 2., 2.],
			[3., 3., 3.]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, of thus 4 weights
# recall that we keep weights transposed
inputs = np.array([[1, 2, 3, 2.5],
			[2., 5., -1., 2],
			[-1.5, 2.7, 3.3, -0.8]])

# sum weigths of given input
# and multiply by the passed in gradiant for this neuron
dweights = np.dot(inputs.T, dvalues)
print(dweights)

# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[1., 1., 1.],
			[2., 2., 2.],
			[3., 3., 3.]])

# One bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plan list -
# we explained this in the chapter 4
dbiases = np.sum(dvalues, axis=0, keepdims=True)
print(dbiases)
