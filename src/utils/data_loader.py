#data.py

#importing lib
#np for computing, keras for data and sklearn for random
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split

#function for getting data bsaed on the arg input from user
def load_data(dataset):
    #x is feature form image and y is o/p number
    if dataset == "MNIST":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset == "FASHION MNIST":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Please give name of data set to be used")

    #setting the data split to 90/10 with randomness of getting random i/p
    x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, test_size=0.1, random_state=69)
    #converting 2d (28, 28) into 1d (784) array
    x_train = x_train.reshape(len(x_train), 784)
    x_val   = x_val.reshape(len(x_val), 784)
    x_test  = x_test.reshape(len(x_test), 784)
    #scaling down the i/p feature in range 0-1
    #using float64 for higher precision float for better accuracy
    x_train = x_train.astype(np.float64) /255
    x_val   = x_val.astype(np.float64)/255
    x_test  = x_test.astype(np.float64) /255

    return x_train, y_train, x_val, y_val, x_test, y_test
