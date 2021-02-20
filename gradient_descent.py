import numpy as np


# Defining the sigmoid function for activations
from IPython.core.display import display


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Input data
x = np.array([0.1, 0.3])
print("x:")
display(x)
# Target
y = 0.2
print("y:")
display(y)
# Input to output weights
weights = np.array([-0.8, 0.5])
print("weights:")
display(weights)

# The learning rate, eta in the weight step equation
learnrate = 0.5
print("learnrate:{}".format(learnrate))

# the linear combination performed by the node (h in f(h) and f'(h))
h = x[0] * weights[0] + x[1] * weights[1]
# or h = np.dot(x, weights)
print("h:{}".format(h))

# The neural network output (y-hat)
nn_output = sigmoid(h)
print("nn_output:{}".format(nn_output))

# output error (y - y-hat)
error = y - nn_output
print("error:{}".format(error))

# output gradient (f'(h))
output_grad = sigmoid_prime(h)
print("output_grad:{}".format(output_grad))

# error term (lowercase delta)
error_term = error * output_grad
print("error_term:{}".format(error_term))

# Gradient descent step
del_w = [learnrate * error_term * x[0],
         learnrate * error_term * x[1]]
# or del_w = learnrate * error_term * x
print("del_w:{}".format(del_w))
