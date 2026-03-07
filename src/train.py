#train.py

#importing arg for customization
import argparse
import numpy as np
import wandb
import random
import json

#importing all the def functions for usage
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
from ann.optimizer import OPT

#defining randomness for wandb
np.random.seed(42)
random.seed(42)

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
    parser.add_argument("-lr","--learning_rate", type=float, required=True, help = "Enter learning rate (η): ")

    #weight decay for L2 reg
    parser.add_argument("-w","--weight_decay", type=float, required=True, help = "Enter weight decaty rate: ")

    #configuration of hidden layer
    parser.add_argument("-nhl","--num_layers", type=int, required=True, help = "Number of Hidden Layers: ")

    #number of neurons in each hidden layer
    parser.add_argument("-sz","--hidden_size", type=int, nargs = '+', required=True, help = "List of number of Neurons in each layer: ")

    #choosing activation function
    parser.add_argument("-a","--activation", type=str, nargs = '+', choices = ["sigmoid", "tanh", "relu" ], required=True, help = "Activation function (sigmoid, tanh, relu): ")

    #chossing initialization
    parser.add_argument("-w_i","--weight_init", type=str, required=True, choices = ["random", "xavier"], help = "W&B initialization (random or xavier): ")

    args = parser.parse_args()

    if len(args.hidden_size) != len(args.activation):
      raise ValueError("There is a mismatch in network size and activation functions")
    return args


#converting all 10 o/p predictions into single guess
def single_guess(y, num_classes=10):
  y_single_guess = np.zeros((len(y), num_classes))
  y_single_guess[np.arange(len(y)), y] = 1
  return y_single_guess


#computing general loss function at o/p layer
def compute_loss(y_true, y_pred, loss_type):
    m = y_true.shape[0]
    if loss_type == "ce":
        y_pred = np.clip(y_pred, 1e-8, 1-1e-8)
        return -np.sum(y_true * np.log(y_pred)) / m

    elif loss_type == "mse":
        return np.mean((y_true - y_pred) ** 2)

    else:
        raise ValueError("Please give name of loss function to be used")


#computing accuray for predictions
def compute_accuracy(y_true, y_pred):
    predict = np.argmax(y_pred, axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(predict == labels)

#training the mlp
def train(model, optimizer, X_train, Y_train, X_val, Y_val, args):

  n = len(X_train)
  step_val = None
  for epoch in range(args.epochs):
      perm = np.random.permutation(n)
      X_train = X_train[perm]
      Y_train = Y_train[perm]

      total_loss = 0
      for i in range(0, len(X_train), args.batch_size):
          batch_slice = slice(i, i + args.batch_size)
          X_batch = X_train[batch_slice]
          Y_batch = Y_train[batch_slice]

          #forward pass
          output, stores = model.forward(X_batch, model.weights, model.biases, model.activations)

          #loss function
          loss = compute_loss(Y_batch, output, args.loss)
          total_loss = total_loss + loss

          #backward pass
          grads_W, grads_b = model.backward(X_batch, Y_batch, model.weights, model.biases, model.activations, args.loss, stores, args.weight_decay)

          if args.optimizer == "sgd":
              model.weights, model.biases = optimizer.sgd(model.weights, model.biases, grads_W, grads_b, args.learning_rate)

          elif args.optimizer == "momentum":
              model.weights, model.biases, step_val = optimizer.momentum(model.weights, model.biases, grads_W, grads_b, args.learning_rate, step_val)

          elif args.optimizer == "nag":
              model.weights, model.biases, step_val = optimizer.nag(model.weights, model.biases, grads_W, grads_b, args.learning_rate, step_val)

          elif args.optimizer == "rmsprop":
              model.weights, model.biases, step_val = optimizer.rmsprop(model.weights, model.biases, grads_W, grads_b, args.learning_rate, step_val)

          elif args.optimizer == "adam":
              model.weights, model.biases, step_val = optimizer.adam(model.weights, model.biases, grads_W, grads_b, args.learning_rate, step_val)

          elif args.optimizer == "nadam":
              model.weights, model.biases, step_val = optimizer.nadam(model.weights, model.biases, grads_W, grads_b, args.learning_rate, step_val)


      val_output, _ = model.forward(X_val, model.weights, model.biases, model.activations)
      print("\nValidation Logits:")
      print(val_output)
      val_accuracy = compute_accuracy(Y_val, val_output)

      print(f"Number of Epochs: {epoch+1}/{args.epochs} | "
            f"Total Loss for batch: {total_loss:.4f} | "
            f"Accuracy of the MLP: {val_accuracy:.4f}")

      wandb.log({"epoch": epoch, "train_loss": total_loss, "val_accuracy": val_accuracy})


#implementing the def function
def main():
    args = parse_args()
    wandb.init(project="DA6401-MLP", config=vars(args))
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(args.dataset)

    Y_train = single_guess(Y_train)
    Y_val = single_guess(Y_val)
    Y_test = single_guess(Y_test)

    #adding softmax automatically for output layer
    activations = args.activation + ["softmax"]
    model = MLP(input_size=784, output_size=10, hidden_sizes=args.hidden_size, activations=activations, weight_init=args.weight_init)
    optimizer = OPT()

    train(model, optimizer, X_train, Y_train, X_val, Y_val, args)
    test_output, _ = model.forward(X_test, model.weights, model.biases, model.activations)
    print("\nTest Logits:")
    print(test_output)

    test_acc = compute_accuracy(Y_test, test_output)
    print(f"Final Accuracy for MLP is: {test_acc:.4f}")
    params = {"W": model.weights, "b": model.biases}
    np.save("../models/model.npy", params)
    print("Model saved successfully as model.npy")

    #saving best configuration
    best_config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "activation": args.activation,
        "weight_init": args.weight_init
    }
    
    with open("best_config.json", "w") as f:
        json.dump(best_config, f, indent=4)
    
    print("Best configuration saved as best_config.json")

if __name__ == "__main__":
    main()
