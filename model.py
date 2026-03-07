#model.py

#importing lib
import numpy as np

#within the class MLP defininf all the functions
class MLP:
  def __init__(self, input_size, output_size, hidden_sizes, activations, weight_init):

    #creating full layer structure: input → hidden → output
    self.layers = [input_size] + hidden_sizes + [output_size]
    #saving activations
    self.activations = activations + [activations[-1]]
    #initializing weights and biases
    self.weights, self.biases = self.initialize_weights(self.layers, weight_init)

  #weights are initialized either with xavier distribution or random val
  def initialize_weights(self, layers, weight_init):
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


  #function for activation function: based on user i/p
  def activation(self, x, activation):
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
  def activation_derivative(self, x, activation):
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
          A = self.activation(Z, activations[i])
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
              dZ = dA * self.activation_derivative(Z_last, activations[-1])

      #for mse the code is same for all act
      elif loss == "mse":
          dA = (A_last - Y)
          dZ = dA * self.activation_derivative(Z_last, activations[-1])
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
              dZ = (dZ @ weights[i].T) * self.activation_derivative(Z_prev, activations[i - 1])
      return grads_W, grads_b


  #prediction
  def predict(self, X, weights, biases, activations):
      probs, _ = self.forward(X, weights, biases, activations)
      return np.argmax(probs, axis=1)
