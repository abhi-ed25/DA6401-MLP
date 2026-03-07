#importing lib
import numpy as np

#objective function for cross entropy
def cross_entropy(Y, A_last):
    #adding a small amount to get rid of non zero div error
    A_last = np.clip(A_last, 1e-8, 1 - 1e-8)
    loss = -np.sum(Y * np.log(A_last)) / Y.shape[0]
    return loss


#objective function for mean squared error
def mse(Y, A_last):
    loss = np.mean((A_last - Y) ** 2)
    return loss
