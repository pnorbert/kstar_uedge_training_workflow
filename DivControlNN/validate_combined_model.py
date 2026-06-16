"""
Script for assessing the combined model performance with an independent validation dataset.

last updated by B. Zhu (zhu12@llnl.gov) 03/11/2025
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
from datetime import datetime
from src.autoencoder import Autoencoder
from src.data import *
from src.diagnose import *

################################################################################
# Configuration Options
################################################################################

# Input data paths
data_path = 'data/'  # Path to raw input data
fname = 'KSTAR_C_high_Ip_val.hdf5'  # File name of input data

# Pre-trained NNs
mmae_path = './models/vae_multimodal_dc_l16_20250311-222854/'  # Path to pre-trained AE model
standardize_z = False  # Whether to standardize latent space representations
mlp_path = './models/mlp_dc_n32_l4_20250311_223941/' # Path to pre-trained MLP model

# Multi-modal autoencoder configuration
arch_name = 'multimodal_dc'  # Autoencoder architecture name
input_shp = ((5, 1), (24, 2), (24, 6))  # Input shape for the autoencoder
feature_sz = (5, 8, 16)  # Feature sizes for the autoencoder
latent_dim = 16  # Latent space dimension
vae = True  # Use Variational Autoencoder (VAE)

# MLP configuration
neurons = 32  # Number of neurons in hidden layers
initial_learning_rate = 1e-2  # Initial learning rate for the optimizer
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
        keras.layers.Dense(8, activation=tf.nn.relu, input_shape=(5,), name='input_pars'),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(neurons, activation=tf.nn.relu),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(neurons, activation=tf.nn.relu),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(latent_dim, activation=None, name='z_mlp_pred')  # Output layer
    ])

    # Compile the model with Adam optimizer and mean squared error loss
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss='mse', optimizer=optimizer)
    return model

################################################################################
# Main Script
################################################################################

if __name__ == '__main__':
    
    # Load pre-trained MLP model for latent space prediction
    mlp = build_mlp_model()
    mlp_weight_path = os.path.join(mlp_path, 'best_val.hdf5') # Path to saved MLP model weight
    #mlp_weight_path = './models/mlp_example2_n32_l4_20250311_182752/best_val.hdf5'  # Path to saved MLP model weights
    #mlp.load_model(mlp_path, compile=False)
    mlp.load_weights(mlp_weight_path)

    # Load autoencoder model
    autoencoder = Autoencoder(
        arch_name, input_shp=input_shp, feature_sz=feature_sz, latent_dim=latent_dim, do_vae=vae
    )
    autoencoder.load_weights(os.path.join(mmae_path, 'weights'))

    # Preprocess validation data
    ip, ncore, pinj, fz, diff = read_data_inputs(data_path, fname)
    ip /= 1000.  # Normalize current
    ncore /= 8.  # Normalize core density
    pinj /= 10.  # Normalize injected power
    fz *= 10.  # Normalize impurity fraction
    diff /= 2.5  # Normalize diffusivity

    # Combine inputs into a single array
    x = np.squeeze(np.stack((ip, ncore, pinj, fz, diff), axis=1))

    # Predict latent space representations using the MLP model
    time_s = datetime.now()
    z = mlp.predict(x)

    # De-standardize latent space representation if needed
    if standardize_z:
        with np.load(os.path.join(mmae_path, 'z.npz')) as data:
            lsr_mean = data['z_mean']
            lsr_std = data['z_std']
        z = lsr_destandardize(z, lsr_mean, lsr_std)

    # Decode latent space representations to physical measurements
    y = autoencoder.decoder(z)

    time_e = datetime.now()
    print("Validation takes ", time_e - time_s)

    # Extract decoded outputs
    y_scal = y[0].numpy()
    y_prf1 = y[1].numpy()
    y_prf2 = y[2].numpy()

    (y_rads, y_teu, y_neu, y_qtl, y_qtr, y_jl, y_jr, y_tel, y_ter) \
            = (np.squeeze(y_scal), y_prf1[:,:,0], y_prf1[:,:,1], \
            y_prf2[:,:,0], y_prf2[:,:,1], y_prf2[:,:,2], \
            y_prf2[:,:,3], y_prf2[:,:,4], y_prf2[:,:,5])

    y_qtl = maxmin_denorm(y_qtl=(y_qtl,(7.5,17.5)))
    y_qtr = maxmin_denorm(y_qtr=(y_qtr,(7.5,17.5)))
    y_jl  = maxmin_denorm(y_jl =(y_jl, (3.5,12.5)))
    y_jr  = maxmin_denorm(y_jr =(y_jr, (3.5,12.5)))
    y_tel = maxmin_denorm(y_tel=(y_tel,(-1.2,5.0)))
    y_ter = maxmin_denorm(y_ter=(y_ter,(-3.5,5.0)))
    y_teu = maxmin_denorm(y_teu=(y_teu,(1.5,7.0)))
    y_neu = maxmin_denorm(y_neu=(y_neu,(41.5,46.0)))
    y_rads[:,0] /= 2.0
    y_rads[:,2] = np.exp(10.0*y_rads[:,2]+10.0)
    y_rads[:,3] /= 2.0
    #y_rads[:,4] = (y_rads[:,4] - 0.45)/20.0
    y_rads[:,4] = (y_rads[:,4] - 0.2)/10.0

    y_qtl = np.squeeze(np.exp(y_qtl))
    y_qtr = np.squeeze(np.exp(y_qtr))
    y_jl  = np.squeeze(np.exp(y_jl))
    y_jr  = np.squeeze(np.exp(y_jr))
    y_tel = np.squeeze(np.exp(y_tel))
    y_ter = np.squeeze(np.exp(y_ter))
    y_teu = np.squeeze(np.exp(y_teu))
    y_neu = np.squeeze(np.exp(y_neu)/1.e19)

    qtl, qtr, jl, jr, tel, ter, teu, neu, rads  = read_data_outputs(data_path,fname)

    # Save combined validation results
    #outpath = './models/combined_validation_results/'
    #os.makedirs(outpath, exist_ok=True)
    outpath = mlp_path # Save data into the MLP model folder
    np.savez(
        os.path.join(outpath, 'combined_validation'),
        qtl = qtl, qtr = qtr, jl = jl, jr =jr, tel = tel,
        ter = ter, teu = teu, neu = neu, rads = rads,
        y_qtl = y_qtl, y_qtr = y_qtr, y_jl = y_jl, y_jr = y_jr,
        y_tel = y_tel, y_ter = y_ter, y_teu = y_teu, y_neu = y_neu,
        y_rads = y_rads
    )

    # Quickly analyze the combined model performance
    plot_combined_validation_example(outpath, 0)
    plot_combined_validation_statistics(outpath)
