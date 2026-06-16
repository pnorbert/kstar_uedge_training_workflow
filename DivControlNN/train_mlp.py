#!/usr/bin/env python3
"""
Script for training an MLP for latent space mapping.

last updated by B. Zhu (zhu12@llnl.gov) 03/11/2025
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
import sys
import datetime
import pickle
from datetime import datetime
import numpy as np
from numpy import *
from src.autoencoder import Autoencoder
from src.data import *
from src.diagnose import *

################################################################################
# Configuration Options
################################################################################

# Training parameters
EPOCHS = 10  # Number of epochs for training
neurons = 32  # Number of neurons in hidden layers
layers = 4  # Number of layers in the MLP
dropout_rate = 0.2  # Dropout rate for regularization
train_split = 0.9  # Fraction of data used for training
batch_size = 256  # Batch size for training
initial_learning_rate = 1e-2  # Initial learning rate for the optimizer
nsample = 72000  # Maximum number of samples to use

# Input data paths
inpath = 'data/'  # Path to raw input data
fname = 'KSTAR_C_high_Ip.hdf5'  # File name of input data

# Latent space representation (z) input path
zinpath = './models/vae_multimodal_dc_l16_20250311-222854/'  # Path to pre-trained VAE model
standardize_z = False  # Whether to standardize latent space representations

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=50000, decay_rate=0.95, staircase=True
)

################################################################################
# Function Definitions
################################################################################

def build_mlp_model():
    """
    Builds a Multi-Layer Perceptron (MLP) model for mapping input parameters
    to latent space representations.
    """
    model = keras.Sequential([
        keras.layers.Dense(8, activation=tf.nn.relu, input_shape=(train_data.shape[1],), name='input_pars'),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(neurons, activation=tf.nn.relu),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(neurons, activation=tf.nn.relu),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(train_labels.shape[1], activation=None, name='z_mlp_pred')  # Output layer
    ])

    # Compile the model with Adam optimizer and mean squared error loss
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss='mse', optimizer=optimizer)
    return model

################################################################################
# Main Script
################################################################################

if __name__ == '__main__':
    # Print TensorFlow version
    print(tf.__version__)

    # Generate a unique timestamp for model naming
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f'mlp_dc_n{neurons}_l{layers}_{ts}'

    # Read and preprocess input data
    ip, ncore, pinj, fz, diff = read_data_inputs(inpath, fname)
    ndata = ncore.shape[0]

    # Normalize input parameters
    ip /= 1000.  # Normalize current
    ncore /= 8.  # Normalize core density
    pinj /= 10.  # Normalize injected power
    fz *= 10.  # Normalize impurity fraction
    diff /= 2.5  # Normalize diffusivity

    # Combine input parameters into a single array
    ips = np.squeeze(np.stack((ip, ncore, pinj, fz, diff), axis=1))

    # Load latent space representation (z) data
    with load(os.path.join(zinpath, 'z.npz')) as data:
        lsr = data['z']

        # Standardize latent space representation if required
        if standardize_z:
            lsr, lsr_mean, lsr_std = lsr_standardize(lsr)

    # Limit the number of samples used
    ndata = min(ndata, nsample)

    print(f'Splitting the data ({train_split} for training)')

    # Split data into training and testing sets
    trn_ids = np.random.choice(np.arange(ndata), int(train_split * ndata), replace=False)
    tst_ids = np.setdiff1d(np.arange(ndata), trn_ids)

    train_data = ips[trn_ids, :]
    train_labels = lsr[trn_ids, :]
    test_data = ips[tst_ids, :]
    test_labels = lsr[tst_ids, :]

    # Build and summarize the MLP model
    model = build_mlp_model()
    model.summary()

    # Early stopping callback to prevent overfitting
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    # Set up model saving path
    outpath = os.path.join('./models', model_name)
    os.makedirs(outpath, exist_ok=True)
    filepath = os.path.join(outpath, 'best_val.hdf5')

    # Model checkpoint callback to save the best model
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto', save_frequency=1, save_format="tf"
    )

    # Start training
    now = datetime.now()
    tstart = now.strftime("%H:%M:%S")
    print("Training starts at ", tstart)

    history = model.fit(
        train_data, train_labels,
        epochs=EPOCHS,
        validation_split=0.2,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[checkpoint]
    )

    # Plot training history
    plot_mlp_training_history(outpath, history)

    now = datetime.now()
    tend = now.strftime("%H:%M:%S")
    print("Training completes at ", tend)

    # Load the best model for evaluation
    model.load_weights(filepath)

    # Predict latent space representations for the test set
    test_pred = model.predict(test_data)

    # Save validation results
    np.savez(os.path.join(outpath, 'mlp_validation'), z_true=test_labels, z_pred=test_pred)
    plot_mlp_validation_statistics(outpath)
