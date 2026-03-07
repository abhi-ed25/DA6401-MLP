#importing lib
import numpy as np

#weights are initialized either with xavier distribution or random val
def initialize_weights(layers, weight_init):

    weights = []
    biases = []

    for i in range(len(layers) - 1):

        #weights connecting to a node: from last and next layer
        #in is number of nodes from last layer
        fan_in = layers[i]

        #out is number of nodes to next layer
        fan_out = layers[i + 1]

        if weight_init == "xavier":

            #using formula for normal xavier initialization
            limit = np.sqrt(2.0 / (fan_in + fan_out))

            #distributing the weights with some randomness for better GD
            W = (np.random.randn(fan_in, fan_out) * limit).astype(np.float64)

        #for random w&b
        elif weight_init == "random":

            #random weight initialization
            W = np.random.randn(fan_in, fan_out).astype(np.float64) * 0.01

        else:
            raise ValueError("Please give name of initialization to be used")

        #initializing biases for all
        b = np.zeros(((1, fan_out)), dtype=np.float64)

        #appending w&b after initialization before training
        weights.append(W)
        biases.append(b)

    return weights, biases
