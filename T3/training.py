#!/usr/bin/env python
from __future__ import division

import os

import numpy as np
import matplotlib.pyplot as plt

from CMAC import CMAC


def plot_loss_curve(loss, dirname, sample_size, receptive):
    plt.plot(loss)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title("Loss Curve, Sample Size: {}, Receptive Field: {}".format(sample_size, receptive))
    plt.grid()
    plt.savefig(os.path.join(dirname, "LossCurve_SampleSize-{}-ReceptiveField-{}.png".format(sample_size, receptive)), dpi=500)
    plt.show()

def save_weights(n_outputs, resolution, weights, output, dirname, foldername='weights'):
    weights = weights.reshape((n_outputs, resolution * resolution))
    np.savetxt(os.path.join(dirname, foldername, "weights.dat"), weights)
    np.savetxt(os.path.join(dirname, foldername, "output.dat"), output)
    return weights.reshape((n_outputs, resolution, resolution))


def Train_CMAC(dataset_filename='train_150.csv'):
    # Load training data
    dirname = os.path.dirname(os.path.abspath(__file__))

    # # Manual training samples
    data = np.loadtxt(os.path.join(dirname, dataset_filename), delimiter=',')

    # read data
    LShoulderPitch = data[:, 0]
    LShoulderRoll = data[:, 1]
    cx = data[:, 2]
    cy = data[:, 3]

    # OUTPUT DATA IS THE LSHOULDER JOINT ANGLES
    y_train = np.vstack((LShoulderPitch, LShoulderRoll))
    y_train = y_train.reshape(2, len(LShoulderPitch))

    X_train = np.vstack((cx, cy))
    X_train = X_train.reshape(2, len(cx))

    # CMAC structure
    n_inputs = X_train.shape[0]
    resolution = 50
    receptive = 5
    n_outputs = y_train.shape[0]

    # Hyperparameters
    epochs = 20
    lr = 0.5

    # Number of training samples & batches
    # should be either 75 or 150
    sample_size = X_train.shape[1]

    # Instantiate neural net class
    model = CMAC(n_inputs, resolution, receptive, n_outputs, epochs, lr)

    weights = np.random.normal(0, 0.1, (n_outputs, resolution, resolution))

    # Init weights
    out = model.random_init_layer2()

    for n in range(n_outputs):
        weights[n] = np.multiply(weights[n], out)


    loss_list = []
    print("\nStart Training...\n")

    for epoch in range(epochs):

        # Shuffle training data each epoch
        np.random.seed(32)
        shuffle_idx = np.random.permutation(sample_size)
        X_train_shuffled, y_train_shuffled = X_train[:, shuffle_idx], y_train[:, shuffle_idx]

        # Init output vector
        output = np.zeros((n_outputs, sample_size))
        loss = 0
        for i in range(sample_size):
            layer2_output, output[:, i] = model.calculate_output(weights, X_train_shuffled[:, i], out)

            # weight update
            weights = model.update_weights(layer2_output, weights, output[:, i], y_train_shuffled[:, i])

            # MSE calculation
            loss += model.calculate_loss(output[:, i], y_train_shuffled[:, i])

        print("Epoch: {}/{}, Training-Loss: {:.4f}".format(epoch + 1, epochs, loss))

        # Append losses after each epoch 
        loss_list.append(loss)

    loss_list = np.vstack(loss_list)

    plot_loss_curve(loss_list, dirname, sample_size, receptive)
    weights = save_weights(n_outputs, resolution, weights, output, dirname)

    return model, weights, out

if __name__ == '__main__':
    # Run CMAC TRAIN
    model, weights, out = Train_CMAC()
