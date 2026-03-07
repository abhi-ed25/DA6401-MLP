#inference.py

#importing arg parse for customization
import argparse
import numpy as np
#importig lib for data
from tensorflow.keras.datasets import mnist, fashion_mnist
#w&b report: automated weights and biases report for all executions
import wandb

#getting the activation for inference
#function for activation function: based on user i/p
def activation(x, activation):
    #activation for forward pass

    if activation == "sigmoid":
        return 1 / (1 + np.exp(-x))
    if activation == "tanh":
        return np.tanh(x)
    if activation == "relu":
        return np.maximum(0, x)
    if activation == "softmax":
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    else:
        raise ValueError("Please give name of activation function to be used")


#arg parse for command line interface
def parse_args():
    parser = argparse.ArgumentParser(description = "Customizing your MLP")

    #chossing data set
    parser.add_argument("-d","--dataset", type=str, required=True, choices = ["MNIST", "FASHION MNIST"], help = "MNIST or FASHION MNIST: ")

    #num of epochs
    parser.add_argument("-e","--epochs", type=int, required=True, help = "Enter number of epochs: ")

    #setting up mini batch size
    parser.add_argument("-b","--batch_size", type=int, required=True, help = "Enter number of data points in  batch: ")

    #chossing loss function
    parser.add_argument("-l","--loss", type=str, required=True, choices = ["mse", "ce"], help = "Loss function (mse or ce): ")

    #setting up mini batch size
    parser.add_argument("-o","--optimizer", type=str, required=True, choices = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help = "Optimizer {sgd, momentum, nag, rmsprop, adam or nadam}: ")

    #eta or learning rate
    parser.add_argument("-lr","--learning_rate", type=float, required=True, help = "Enter learning rate (eta): ")

    #weight decay for L2 reg
    parser.add_argument("-w","--weight_decay", type=float, required=True, help = "Enter weight decaty rate: ")

    #configuration of hidden layer
    parser.add_argument("-nhl","--num_layers", type=int, required=True, help = "Number of Hidden Layers: ")

    #number of neurons in each hidden layer, nargs to seperate the elements of list
    parser.add_argument("-sz","--hidden_size", type=int, nargs = '+', required=True, help = "List of number of Neurons in each layer: ")

    #choosing activation function, nargs to seperate the elements of list
    parser.add_argument("-a","--activation", type=str, nargs = '+', choices = ["sigmoid", "tanh", "relu" ], required=True, help = "Activation function (sigmoid, tanh, relu): ")

    #chossing initialization
    parser.add_argument("-w_i","--weight_init", type=str, required=True, choices = ["random", "xavier"], help = "W&B initialization (random or xavier): ")

    args = parser.parse_args()

    #here the number of hidden layers shall match choices for activation
    if len(args.hidden_size) != len(args.activation):
      raise ValueError("Mismatch in network size and choices for activation")
    else:
      pass

    return args



#loading mnist for fashion data
def load_data(dataset):
  #only loading test data as training is done
  if dataset == "MNIST":
      (_, _), (x_test, y_test) = mnist.load_data()

  elif dataset == "FASHION MNIST":
      (_, _), (x_test, y_test) = fashion_mnist.load_data()

  else:
      raise ValueError("Please give name of data set to be used")


  #converting 2d (28, 28) into 1d array
  x_test  = x_test.reshape(x_test.shape[0], -1)
  #normalize i/p feature from 0-255 to 0-1 with max accuracy
  x_test  = x_test.astype(np.float64) / 255.0
  return x_test, y_test


#forward pass: similar to training in which model.py had similar code
def forward(X, weights, biases, activations):
    A = X
    #gidden layer pass
    for i in range(len(weights) - 1):
        Z = A @ weights[i] + biases[i]
        A = activations[i](Z)
    #output activation is last in list of arg parse
    Z = A @ weights[-1] + biases[-1]
    A = activations[-1](Z)
    return A



#performance metrics for mlp: Accuracy, Precision, Recall, and F1-score
def compute_metrics(y_true, y_pred):
  #accuracy
  accuracy = np.mean(y_true == y_pred)

  #precision, recall, f1
  num_classes = 10
  precision_list = []
  recall_list = []
  f1_list = []

  #computing these metrics for all classes of mnist
  for c in range(num_classes):
    #computing true positive, false positive and false negative for all cases
    tp = np.sum((y_pred == c) & (y_true == c))
    fp = np.sum((y_pred == c) & (y_true != c))
    fn = np.sum((y_pred != c) & (y_true == c))

    #calculating precesion, recall and f1 score based on these values
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)


    #storing all the results in a list for showcasing preogression of scores
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

  return (accuracy, np.mean(precision_list), np.mean(recall_list), np.mean(f1_list))


#executing all the def functions
def main():
  #allowing inout from user via cli
  args = parse_args()
  #loading data
  print("Please wait: loading test data...")
  X_test, y_test = load_data(args.dataset)
  #initilizing and loading
  print("Please wait: loading model weights...")
  params = np.load("model.npy", allow_pickle=True).item()
  weights = [w.astype(np.float64) for w in params["W"]]
  biases  = [b.astype(np.float64) for b in params["b"]]
  #running all the arg parse for delivering values and commands
  activation_functions = [lambda x, act=a: activation(x, act) for a in args.activation]
  #output layer activation is always softmax
  activation_functions.append(lambda x: activation(x, "softmax"))
  #forward pass for inference
  print("Please wait: running inference...")
  outputs = forward(X_test, weights, biases, activation_functions)
  #predicitons from mlp
  predictions = np.argmax(outputs, axis=1)
  #calculating the performance metric for the case
  acc, prec, rec, f1 = compute_metrics(y_test, predictions)

  #saving w&b report
  wandb.init(project="DA6401-MLP-Inference", config=vars(args))

  #logging in the metrics
  wandb.log({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})
  wandb.finish()

  #print output of the inference run
  print("Evaluation Results are as follows")
  print("  ")
  print(f"Accuracy  : {acc * 100:.2f}%")
  print(f"Precision : {prec:.4f}")
  print(f"Recall    : {rec:.4f}")
  print(f"F1-score  : {f1:.4f}")



#running inference
if __name__ == "__main__":
    main()
