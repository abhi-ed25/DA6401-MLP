# optimizers.py
import numpy as np

#similar to MLP class OPT class stores all the optimizers
class OPT:
  #stochastic gd
  def sgd(self, weights, biases, grads_W, grads_b, lr):
    #gradient descent for all weights and bias
    for i in range(0, len(weights)):
      #new weight  = old + gradient
      weights[i] = weights[i] - (lr * grads_W[i])
      biases[i]  = biases[i] - (lr * grads_b[i])
    return weights, biases

  #momentum based gd: decay value is set to 0.9
  def momentum(self, weights, biases, grads_W, grads_b, lr, step_val, beta=0.9):
    #initializing the steps with corrections as zero
    if step_val is None:
        step_val = {"svw": [np.zeros_like(w, dtype=np.float64) for w in weights], "svb": [np.zeros_like(b, dtype=np.float64) for b in biases]}
    #computing the correction factor for all weights and biases and adding the into original value
    for i in range(0, len(weights)):
      #"momentum" is calculted at each step and added to weight or bias
      step_val["svw"][i] = beta * step_val["svw"][i] + grads_W[i]
      weights[i] = weights[i] - (lr * step_val["svw"][i])
      step_val["svb"][i] = beta * step_val["svb"][i] + grads_b[i]
      biases[i]  = biases[i] - (lr * step_val["svb"][i])
    return weights, biases, step_val


  #nag optimizer: with set decay value of 0.9
  def nag(self, weights, biases, grads_W, grads_b, lr, step_val, beta=0.9):
    if step_val is None:
        step_val = {"svw": [np.zeros_like(w, dtype=np.float64) for w in weights], "svb": [np.zeros_like(b, dtype=np.float64) for b in biases]}
    #similar kind of initialization as momentum
    #acceleration has partial value of previous grad + current gradient
    for i in range(0, len(weights)):
      #last value of velocity, if not then initial value
      prev_svw = step_val["svw"][i]
      #velocity update
      step_val["svw"][i] = beta * step_val["svw"][i] + grads_W[i]
      #weight update
      weights[i] = weights[i] - lr * (beta * prev_svw + grads_W[i])
      #same for the bias as well
      prev_svb = step_val["svb"][i]
      step_val["svb"][i] = beta * step_val["svb"][i] + grads_b[i]
      #bias update
      biases[i] = biases[i] - lr * (beta * prev_svb + grads_b[i])
    return weights, biases, step_val


  #rmsprop: with decay of 0.9 and epslon as very small non zero value
  def rmsprop(self, weights, biases, grads_W, grads_b, lr, step_val, beta=0.9, eps=1e-8):
    #zero initialization
    if step_val is None:
        step_val = {"svw": [np.zeros_like(w, dtype=np.float64) for w in weights], "svb": [np.zeros_like(b, dtype=np.float64) for b in biases]}
    #computing grad at each step and adding it into w&b
    for i in range(0, len(weights)):
      #rms calculation
      step_val["svw"][i] = beta * step_val["svw"][i] + (1 - beta) * (grads_W[i] ** 2)
      #rms propagation
      weights[i] = weights[i] - (lr * grads_W[i] / (np.sqrt(step_val["svw"][i]) + eps))
      #smae for biases
      step_val["svb"][i] = beta * step_val["svb"][i] + (1 - beta) * (grads_b[i] ** 2)
      biases[i]  = biases[i] - (lr * grads_b[i] / (np.sqrt(step_val["svb"][i]) + eps))
    return weights, biases, step_val


  #adam optimizer: similar eplson as small values (non zero)
  #beta 1 and 2 are the decay rate of first and second moment
  def adam(self, weights, biases, grads_W, grads_b, lr, step_val, beta1=0.9, beta2=0.999, eps=1e-8):
    if step_val is None:
      #split in two lines coz code was too long
      step_val = {"t": 0, "mw": [np.zeros_like(w, dtype=np.float64) for w in weights], "vw": [np.zeros_like(w, dtype=np.float64) for w in weights],
                  "mb": [np.zeros_like(b, dtype=np.float64) for b in biases], "vb": [np.zeros_like(b, dtype=np.float64) for b in biases]}
    #time step: t
    step_val["t"] += 1
    t = step_val["t"]
    #loop for adam opt with both moments for all weights
    for i in range(0, len(weights)):
      #first moment for adam
      step_val["mw"][i] = beta1 * step_val["mw"][i] + (1 - beta1) * grads_W[i]
      #second moment for adam
      step_val["vw"][i] = beta2 * step_val["vw"][i] + (1 - beta2) * (grads_W[i] ** 2)
      #bias correction
      mw_hat = step_val["mw"][i] / (1 - beta1 ** t)
      vw_hat = step_val["vw"][i] / (1 - beta2 ** t)
      #update: weight
      weights[i] -= lr * mw_hat / (np.sqrt(vw_hat) + eps)

      #first momentum for bias
      step_val["mb"][i] = beta1 * step_val["mb"][i] + (1 - beta1) * grads_b[i]
      #second moment for bias
      step_val["vb"][i] = beta2 * step_val["vb"][i] + (1 - beta2) * (grads_b[i] ** 2)
      #bias correction
      mb_hat = step_val["mb"][i] / (1 - beta1 ** t)
      vb_hat = step_val["vb"][i] / (1 - beta2 ** t)
      #update: bias
      biases[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
    return weights, biases, step_val


  #nadam: first moment decay is 0.9 and second moment decay rate is 0.999
  def nadam(self, weights, biases, grads_W, grads_b, lr, step_val, beta1=0.9, beta2=0.999, eps=1e-8):
    #same initialization as adam
    if step_val is None:
        step_val = {"t": 0, "mw": [np.zeros_like(w, dtype=np.float64) for w in weights], "vw": [np.zeros_like(w, dtype=np.float64) for w in weights],
                    "mb": [np.zeros_like(b, dtype=np.float64) for b in biases], "vb": [np.zeros_like(b, dtype=np.float64) for b in biases]}
    #time step: t
    step_val["t"] += 1
    t = step_val["t"]

    for i in range(0, len(weights)):
      #first moment for weight
      step_val["mw"][i] = beta1 * step_val["mw"][i] + (1 - beta1) * grads_W[i]
      #second moment for weight
      step_val["vw"][i] = beta2 * step_val["vw"][i] + (1 - beta2) * (grads_W[i] ** 2)

      #bias correction
      mw_hat = step_val["mw"][i] / (1 - beta1 ** t)
      vw_hat = step_val["vw"][i] / (1 - beta2 ** t)
      #Nesterov correction
      grad_hat_W = grads_W[i] / (1 - beta1 ** t)
      m_nesterov_W = beta1 * mw_hat + (1 - beta1) * grad_hat_W
      #update: weight
      weights[i] -= lr * m_nesterov_W / (np.sqrt(vw_hat) + eps)


      #first moment for bias
      step_val["mb"][i] = beta1 * step_val["mb"][i] + (1 - beta1) * grads_b[i]
      #second moment for bias
      step_val["vb"][i] = beta2 * step_val["vb"][i] + (1 - beta2) * (grads_b[i] ** 2)
      #bias correction
      mb_hat = step_val["mb"][i] / (1 - beta1 ** t)
      vb_hat = step_val["vb"][i] / (1 - beta2 ** t)
      #Nesterov correction
      grad_hat_b = grads_b[i] / (1 - beta1 ** t)
      m_nesterov_b = beta1 * mb_hat + (1 - beta1) * grad_hat_b
      #update: bias
      biases[i] -= lr * m_nesterov_b / (np.sqrt(vb_hat) + eps)
    return weights, biases, step_val
