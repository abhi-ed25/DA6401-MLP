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
  wandb.log({
      "accuracy": acc,
      "precision": prec,
      "recall": rec,
      "f1_score": f1
  })
  wandb.finish()

  #print output of the inference run
  print("Evaluation Results are as follows")
  print("  ")
  print(f"Accuracy  : {acc * 100:.2f}%")
  print(f"Precision : {prec:.4f}")
  print(f"Recall    : {rec:.4f}")
  print(f"F1-score  : {f1:.4f}")




#sweep for the hyper parameter 100 test in 2.2
def sweep_run():
    wandb.init(project="DA6401-MLP-Inference")
    config = wandb.config

    #loading the data for mnist digits
    if config.dataset == "MNIST":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    #reshaping and normalize
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float64) / 255.0
    x_test  = x_test.reshape(x_test.shape[0], -1).astype(np.float64) / 255.0

    #creating architecture from sweep config
    input_size = x_train.shape[1]
    output_size = 10
    hidden_sizes = [config.hidden_size] if isinstance(config.hidden_size, int) else config.hidden_size
    layer_sizes = [input_size] + hidden_sizes + [output_size]

    weights = []
    biases = []

    #initialization from sweep config
    for i in range(len(layer_sizes) - 1):
        if config.weight_init == "xavier":
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            W = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
        else:
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
        b = np.zeros((1, layer_sizes[i+1]))
        weights.append(W)
        biases.append(b)

    #activation functions
    activation_list = [config.activation] * len(weights)
    activation_functions = [lambda x, act=a: activation(x, act) for a in activation_list]

    #training accuracy (no training, just evaluating random init)
    train_outputs = forward(x_train, weights, biases, activation_functions)
    print("\nLogits (Forward Pass Output):")
    print(train_outputs)
    train_preds = np.argmax(train_outputs, axis=1)
    train_acc = np.mean(train_preds == y_train)

    #testing accuracy
    test_outputs = forward(x_test, weights, biases, activation_functions)
    print("\nTest Logits:")
    print(test_outputs)
    test_preds = np.argmax(test_outputs, axis=1)
    test_acc = np.mean(test_preds == y_test)

    #logging the data for both
    wandb.log({"accuracy": test_acc, "train_accuracy": train_acc, "test_accuracy": test_acc})

    #creating table for overlay visualization
    overlay_table = wandb.Table(columns=["metric", "accuracy"])
    overlay_table.add_data("Train Accuracy", train_acc)
    overlay_table.add_data("Test Accuracy", test_acc)

    wandb.log({"Train vs Test Accuracy Overlay":
               wandb.plot.line(overlay_table, "metric", "accuracy",
                               title="Overlay Plot: Train vs Test Accuracy")})

    wandb.finish()


#possible hyp param combinations for the sweep
#creating dictionary for all permutations
sweep_config = {
    "method": "random",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "dataset": {"values": ["MNIST"]},
        "epochs": {"values": [1, 10, 20]},
        "batch_size": {"values": [16, 64, 256]},
        "loss": {"values": ["mse", "ce"]},
        "optimizer": {"values": ["sgd", "adam", "nadam"]},
        "learning_rate": {"values": [0.01, 0.05, 0.1]},
        "weight_decay": {"values": [0.0001, 0.01, 0.1]},
        "num_layers": {"values": [1, 3, 6]},
        "hidden_size": {"values": [64, 128]},
        "activation": {"values": ["sigmoid", "relu"]},
        "weight_init": {"values": ["xavier", "random"]}
    }
}

#comparison between xavier and zero initialization
def zero(args):
    wandb.init(project="DA6401-MLP-Zero-vs-Xavier", config=vars(args))
    #loading data (only one batch needed for gradient visualization)
    if args.dataset == "MNIST":
        (x_train, y_train), _ = mnist.load_data()
    else:
        (x_train, y_train), _ = fashion_mnist.load_data()

    #reshaping and normalize
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float64) / 255.0
    #taking one mini batch
    X = x_train[:args.batch_size]
    y = y_train[:args.batch_size]

    input_size = X.shape[1]
    output_size = 10
    hidden_sizes = args.hidden_size
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    gradients_xavier = []
    gradients_zero = []

    for init_type in ["xavier", "zero"]:
        weights = []
        biases = []

        for i in range(len(layer_sizes) - 1):
            if init_type == "xavier":
                limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
                W = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            else:
                 W = np.zeros((layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))
            weights.append(W)
            biases.append(b)

        #forward pass storing activations
        A = X
        activations_store = [A]
        activation_functions = [lambda x, act=a: activation(x, act) for a in args.activation]

        for i in range(len(weights) - 1):
            Z = A @ weights[i] + biases[i]
            A = activation_functions[i](Z)
            activations_store.append(A)

        #output layer (softmax-like gradient assumption using CE derivative)
        Z = A @ weights[-1] + biases[-1]
        exp_scores = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(y)), y] = 1

        dZ = probs - one_hot
        #backprop to first hidden layer
        dW_last = activations_store[-1].T @ dZ
        dA_prev = dZ @ weights[-1].T

        #first hidden layer gradient
        if args.activation[0] == "relu":
            dZ_hidden = dA_prev * (activations_store[1] > 0)
        elif args.activation[0] == "sigmoid":
            sig = activations_store[1]
            dZ_hidden = dA_prev * sig * (1 - sig)
        else:
            dZ_hidden = dA_prev * (1 - activations_store[1]**2)

        dW_first = activations_store[0].T @ dZ_hidden

        #compute gradient norm per neuron (columns correspond to neurons)
        grad_norm = np.linalg.norm(dW_first, axis=0)

        if init_type == "xavier":
            gradients_xavier = grad_norm
        else:
            gradients_zero = grad_norm

    #logging plots separately
    table_x = wandb.Table(data=[[i, gradients_xavier[i]] for i in range(len(gradients_xavier))],
                          columns=["neuron_index", "gradient_norm"])
    table_z = wandb.Table(data=[[i, gradients_zero[i]] for i in range(len(gradients_zero))],
                          columns=["neuron_index", "gradient_norm"])
    wandb.log({
        "Xavier Initialization Gradient (Layer 1)": wandb.plot.line(table_x, "neuron_index", "gradient_norm", title="Gradient Norms - Xavier Initialization"),
        "Zero Initialization Gradient (Layer 1)": wandb.plot.line(table_z, "neuron_index", "gradient_norm", title="Gradient Norms - Zero Initialization")})
    wandb.finish()


#run the sweep for all 100 combination
if __name__ == "__main__":
  import sys
  #if inferenceis ran for param sweep
  if "--sweep" in sys.argv:
    sweep_id = wandb.sweep(sweep_config, project="DA6401-MLP-Inference")
    wandb.agent(sweep_id, sweep_run, count=100)
  #2.9 zero problem
  elif "--zero" in sys.argv:
    args = parse_args()
    zero(args)
  #if inference is ran for a certain case
  else:
    main()
