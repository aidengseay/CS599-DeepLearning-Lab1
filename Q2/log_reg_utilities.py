################################################################################
# Logistic Regression Utilities
# Aiden Seay, CS 599 - Deep Learning - Fall 2025
################################################################################
# Imports

import time
import tensorflow as tf
from mnist_reader import *
import matplotlib.pyplot as plt

################################################################################
# Constants

SGD = 100
ADAM = 101
ADAGRAD = 102
NADAM = 103
RMSPROP = 104


################################################################################
# Model Function

def run_model(learning_rate,
              input_batch_size,
              n_epochs,
              val_fract,
              optimizer_type):

    """
    Definition: runs the logistic regression model.

    Inputs: valid_fract (float) - the fraction in decimal form of the proportion
                                  of training data that is validation data.

    Outputs: history of the model
    """

    # read in the data
    X_val, y_val, X_train, y_train, X_test, y_test = get_data(val_fract)

    # define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape = (X_train.shape[1],)),
        tf.keras.layers.Dense(units = 10, activation = "sigmoid")
    ])

    # compile the model
    model.compile(
        optimizer = get_optimizer(optimizer_type, learning_rate),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = ["accuracy"]
    )

    # train the model
    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        batch_size = input_batch_size,
        epochs= n_epochs,
    )

    return model, history


################################################################################
# Supporting Functions

def get_data(valid_fract):

    """
    Definition: gets training, validation, and testing datasets from external
                source. Additionally normalizes the data and ensures datatypes
                are correct for training.

    Inputs: valid_fract (float) - the fraction in decimal form of the proportion
                                  of training data that is validation data.

    Outputs: X_val (tensor) - the features in validation set
             y_val (tensor) - the results in the validation set
             X_train (tensor) - the features in the training set
             y_train (tensor) - the results in the training set
             X_test (tensor) - the features in the testing set
             y_test (tensor) -  the results in the testing set
    """

    # read in the data
    fmnist_folder = "."
    X_train_all, y_train_all = load_mnist(fmnist_folder, kind='train')
    X_test, y_test = load_mnist(fmnist_folder, kind='t10k')

    # compute fraction of training data being used for validation
    num_val = int(len(X_train_all) * valid_fract)

    # shuffle to ensure random validation set
    indices = np.arange(len(X_train_all))
    np.random.shuffle(indices)

    X_train_all = X_train_all[indices]
    y_train_all = y_train_all[indices]

    # split training data into training and validation subsets
    X_val = X_train_all[:num_val]
    y_val = y_train_all[:num_val]
    X_train = X_train_all[num_val:]
    y_train = y_train_all[num_val:]

    # ensure all values are flt or int for later computations
    # also need to normalize the features between 0 - 1
    X_val = X_val.astype("float32") / 255.0
    y_val = y_val.astype("int32")
    X_train = X_train.astype("float32") / 255.0
    y_train = y_train.astype("int32")
    X_test = X_test.astype("float32") / 255.0
    y_test = y_test.astype("int32")

    # display train, validation, and test split for check
    print(f"Train Set:        {X_train.shape}, {y_train.shape}")
    print(f"Validation Set:   {X_val.shape} , {y_val.shape}")
    print(f"Test Set :        {X_test.shape}, {y_test.shape}")

    return X_val, y_val, X_train, y_train, X_test, y_test


def get_optimizer(optimizer_type, input_learning_rate):

    """
    Definition: gets the specific optimizer used for a model.

    Inputs: optimizer_type (int) - the constant used to define what optimizer is
                                   used.

    Outputs: optimizer function - the function used to optimize the model.
    """

    if optimizer_type == SGD:

        return tf.keras.optimizers.SGD(learning_rate=input_learning_rate)

    elif optimizer_type == ADAM:

        return tf.keras.optimizers.Adam(learning_rate=input_learning_rate)
    
    elif optimizer_type == ADAGRAD:

        return tf.keras.optimizers.Adagrad(learning_rate=input_learning_rate)

    elif optimizer_type == RMSPROP:

        return tf.keras.optimizers.RMSprop(learning_rate=input_learning_rate)
    
    # default to NADAM if no other conditions are true
    else:

        return tf.keras.optimizers.Nadam(learning_rate=input_learning_rate)
    
################################################################################
# Supporting Plotting Functions (Given)

def plot_images(images, y, yhat=None):
    assert len(images) == len(y) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape((28, 28)), cmap='binary')

        # Show true and predicted classes.
        if yhat is None:
            xlabel = "True: {0}".format(y[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(y[i], yhat[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def plot_weights(model):

    # Extract weights
    w, b = model.layers[0].get_weights()

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i < 10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape((28, 28))

            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
            
            # Set the label for the sub-plot.
            ax.set_xlabel(f"Weights: {i}")
        
        # Remove ticks from each sub-plot.
        ax.set_xticks([]); ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

################################################################################