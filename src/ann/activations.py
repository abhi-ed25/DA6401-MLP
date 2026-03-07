#importing lib
import numpy as np

#function for activation function: based on user i/p
def activation(x, activation):
    #activation for forward pass
    #activation is i/p from arg parse
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif activation == "tanh":
        return np.tanh(x)
    elif activation == "relu":
        return np.maximum(0, x)
    else:
        raise ValueError("Please give name of activation function to be used")


#function for derivative
def activation_derivative(x, activation):
    #derivative for backward pass
    if activation == "sigmoid":
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)
    elif activation == "tanh":
        return 1 - np.tanh(x) ** 2
    elif activation == "relu":
        return (x > 0).astype(float)
    else:
        raise ValueError("Please give name of activation function to be used")
