#!/usr/bin/env python3
"""
Script for training an autoencoder for detachment control.

original by B. Zhu (zhu12@llnl.gov) 03/11/2025
rewritten to read from downloaded .bp adios2 files
"""

import os
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from pathlib import Path
from DivControlNN.src.autoencoder import Autoencoder
from DivControlNN.src.data import *
from DivControlNN.src.diagnose import *

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Enable eager execution for TensorFlow
tf.config.run_functions_eagerly(True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ------------------------------------------------------------------------------
# Configuration: Input/Output Dimensions, Training Parameters, and Paths
# ------------------------------------------------------------------------------

# Input dimensions (hard-coded for detachment control example)
input_sz = [(24,), (24,), (24,), (24,), (24,), (24,), (24,), (24,), (5,)]
latent_dim = 16  # Latent space dimensionality

# Model architecture and training parameters
arch_name = 'multimodal_dc'
nepochs = 10
train_split = 0.9
valid_split = 0.2
batch_size = 256
initial_learning_rate = 2e-2
nsample = 72000
vae = True  # Use Variational Autoencoder (VAE)

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.95, staircase=True
)

# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------

def train_autoencoder(inpath: Path, model_id: str):

    # Define model name based on whether VAE is used
    model_name = f'vae_{arch_name}_l{latent_dim}_{model_id}' if vae else f'ae_{arch_name}_l{latent_dim}_{model_id}'

    # Read and preprocess data
    qtl, qtr, jl, jr, tel, ter, teu, neu, rads = read_data_outputs(str(inpath / "training_set.bp"))
    ndata = rads.shape[0]

    # Validate input sizes
    assert input_sz[0] == qtl.shape[1:]
    assert input_sz[1] == qtr.shape[1:]
    assert input_sz[2] == jl.shape[1:]
    assert input_sz[3] == jr.shape[1:]
    assert input_sz[4] == tel.shape[1:]
    assert input_sz[5] == ter.shape[1:]
    assert input_sz[6] == teu.shape[1:]
    assert input_sz[7] == neu.shape[1:]
    assert input_sz[8] == rads.shape[1:]

    # Normalize and preprocess data for neural network training
    qtl = np.log(np.abs(qtl))
    qtr = np.log(qtr)
    jl = np.log(jl)
    jr = np.log(jr)
    tel = np.log(tel)
    ter = np.log(ter)
    teu = np.log(teu)
    neu = np.log(neu * 1e19)

    # Apply normalization for May 2024 KSTAR data
    (qtl, qtl_mm) = maxmin_norm(qtl, (7.5, 17.5))
    (qtr, qtr_mm) = maxmin_norm(qtr, (7.5, 17.5))
    (jl, jl_mm) = maxmin_norm(jl, (3.5, 12.5))
    (jr, jr_mm) = maxmin_norm(jr, (3.5, 12.5))
    (tel, tel_mm) = maxmin_norm(tel, (-1.2, 5.0))
    (ter, ter_mm) = maxmin_norm(ter, (-3.5, 5.0))
    (teu, teu_mm) = maxmin_norm(teu, (1.5, 7.0))
    (neu, neu_mm) = maxmin_norm(neu, (41.5, 46.0))
    rads[:, 0] *= 2.0
    rads[:, 2] = 0.1 * (np.log(rads[:, 2]) - 10.0)
    rads[:, 3] *= 2.0
    rads[:, 4] = 10.0 * rads[:, 4] + 0.2

    # Split data into training, validation, and testing sets
    ndata = min(ndata, nsample)
    print(f'Splitting the data ({train_split} for training and validation)')
    tst_ids = np.random.choice(np.arange(ndata), int((1 - train_split) * ndata), replace=False)
    trn_ids = np.setdiff1d(np.arange(ndata), tst_ids)
    val_ids = np.random.choice(trn_ids, int(valid_split * len(trn_ids)), replace=False)
    trn_ids = np.setdiff1d(trn_ids, val_ids)
    np.random.shuffle(trn_ids)

    print(f'Training data: {trn_ids.shape}, Validation data: {val_ids.shape}, Testing data: {tst_ids.shape}')

    # Prepare data for the autoencoder
    print(f'Multi-modal autoencoder is required for this example.')
    dscalars = rads
    dprofiles1 = np.stack((teu, neu), axis=-1)
    dprofiles2 = np.stack((qtl, qtr, jl, jr, tel, ter), axis=-1)

    train_data = (dscalars[trn_ids], dprofiles1[trn_ids], dprofiles2[trn_ids])
    val_data = (dscalars[val_ids], dprofiles1[val_ids], dprofiles2[val_ids])
    test_data = (dscalars[tst_ids], dprofiles1[tst_ids], dprofiles2[tst_ids])
    full_data = (dscalars, dprofiles1, dprofiles2)

    print(f'Combined data: scalars = {dscalars.shape}, profiles1 = {dprofiles1.shape}, profiles2 = {dprofiles2.shape}')

    # Define input and feature sizes
    scal_sz = input_sz[8] + (1,)
    prf1_sz = input_sz[6] + (2,)
    prf2_sz = input_sz[0] + (6,)
    input_shp = (scal_sz, prf1_sz, prf2_sz)
    feature_sz = (5, 8, 16)

    # Initialize the autoencoder
    autoencoder = Autoencoder(
        arch_name,
        input_shp=input_shp,
        feature_sz=feature_sz,
        latent_dim=latent_dim,
        do_vae=vae
    )

    # Create output directory
    outpath = os.path.join('./models', model_name)
    os.makedirs(outpath, exist_ok=True)
    autoencoder.write_model_summary(outpath)

    # Start training
    now = datetime.now()
    tstart = now.strftime("%H:%M:%S")
    print("Training starts at ", tstart)

    autoencoder.train(train_data, val_data, nepochs, batch_size, lr_schedule, outpath)

    now = datetime.now()
    tend = now.strftime("%H:%M:%S")
    print("Training completes at ", tend)

    # Save training results and visualizations
    plot_ae_training_history(outpath, 'mmvae' if vae else 'mmae')

    autoencoder.save_weights(os.path.join(outpath, 'weights'))

    # Evaluate the model on the test set
    with tf.device('/cpu:0'):
        z = autoencoder.encoder(test_data)
        pred_data = autoencoder.decoder(z)
        test_scal = test_data[0]
        test_prf1 = test_data[1]
        test_prf2 = test_data[2]
        pred_scal = pred_data[0].numpy()
        pred_prf1 = pred_data[1].numpy()
        pred_prf2 = pred_data[2].numpy()

        # Save validation results
        np.savez(
            os.path.join(outpath, 'validation'),
            test_scal=test_scal,
            test_prf1=test_prf1,
            test_prf2=test_prf2,
            pred_scal=pred_scal,
            pred_prf1=pred_prf1,
            pred_prf2=pred_prf2
        )

        plot_ae_validation_example(outpath, 0)
        plot_ae_validation_statistics(outpath)

    # Save latent space representation
    z = autoencoder.encoder(full_data)
    _, z_mean, z_std = lsr_standardize(z)  # Ignoring the first output as it's not needed (for now)
    np.savez(os.path.join(outpath, 'z'), z=z, z_mean=z_mean, z_std=z_std)
    plot_lsr_distribution(outpath)
