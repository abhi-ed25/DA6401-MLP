#importing lib
import numpy as np
from ann.activations import activation, activation_derivative
from ann.neural_layer import initialize_weights


#within the class MLP defininf all the functions
class MLP:
  def __init__(self, input_size, output_size, hidden_sizes, activations, weight_init):
    #creating full layer structure: input → hidden → output
    self.layers = [input_size] + hidden_sizes + [output_size]
    #saving activations
    self.activations = activations + [activations[-1]]
    #initializing weights and biases
    self.weights, self.biases = initialize_weights(self.layers, weight_init)


  #initializing a forward pass
  def forward(self, X, weights, biases, activations):
      # A is set to be i/p feature for further use
      A = X

      #stored in a list of stores
      stores = []

      #forward computation at all layers
      for i in range(len(weights)):
          # @ used for summation of all w.x
          Z = A @ weights[i] + biases[i]
          A = activation(Z, activations[i])

          #appending forward pass
          stores.append((A, Z))
      return A, stores


  #backward pass or learning
  def backward(self, X, Y, weights, biases, activations, loss, stores, weight_decay):

      #empty list for gradients for w&b
      grads_W = []
      grads_b = []

      self.grad_W = grads_W
      self.grad_b = grads_b
      m = X.shape[0]

      #o/p of forward pass at last node is prediction this is declared here
      A_last, Z_last = stores[-1]

      #gradient for output layer
      if loss == "ce":
          #cross entropy gradient only for sigmoid is simple computation
          if activations[-1] == "sigmoid":
              dZ = A_last - Y

          else:
              #ce plus others need expansion terms which are here
              dA = -(Y / (A_last + 1e-8)) + ((1 - Y) / (1 - A_last + 1e-8))
              #adding a small amount to get rid of non zero div error
              dZ = dA * activation_derivative(Z_last, activations[-1])

      #for mse the code is same for all act
      elif loss == "mse":
          dA = (A_last - Y)
          dZ = dA * activation_derivative(Z_last, activations[-1])

      else:
          raise ValueError("Please give loss function to be used")


      #updating weights and biases
      for i in reversed(range(len(weights))):
          A_prev = X if i == 0 else stores[i - 1][0]

          #gradients for entire weight
          dW = (A_prev.T @ dZ) / m + weight_decay * weights[i]

          #gradients for entire bias
          db = np.sum(dZ, axis=0, keepdims=True) / m

          #final value of the gradient
          grads_W.insert(0, dW)
          grads_b.insert(0, db)

          #storing the vals for all
          if i != 0:
              _, Z_prev = stores[i - 1]
              dZ = (dZ @ weights[i].T) * activation_derivative(Z_prev, activations[i - 1])
      return grads_W, grads_b


  #prediction
  def predict(self, X, weights, biases, activations):
      probs, _ = self.forward(X, weights, biases, activations)
      return np.argmax(probs, axis=1)
