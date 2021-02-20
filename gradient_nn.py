import numpy as np
from IPython.core.display import display


def sigmoid(x):
    """
    Calculate sigmoid
    :param x:
    :return: sigmoid value
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """
    Derivative of the sigmoid function
    :param x:
    :return: return the sigmoid derivative
    """
    return sigmoid(x) * (1 - sigmoid(x))

learnrate = 0.5
x = np.array([1,2,3,4])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5, 0.3, 0.1])

### Calculate one gradient descent step for each weight
### Note: Some steps have been consolidated, so there are
###       fewer variable names than in the above sample code

# Calculate the node's linear combination of inputs and weights
h = np.dot(x, w)

# Calculate output of neural network
nn_output = sigmoid(h)

# Calculate the error of the neural network
error = y - nn_output

# Calculate the error term
# Remember, this requires output gradient, which we haven't
# specifically added a variable for.
error_term = error * sigmoid_prime(h)

# Calculate change in weights
del_w = learnrate * error_term * x

print("x:")
display(x)
print("y:")
display(y)
print("w:")
display(w)
print("Neural Network output:")
display(nn_output)
print("Amount of Error:")
display(error)
print("Change in Weights:")
display(del_w)
