# ------------------------------------------------------------------------------
# Utility Functions for Plotting and Analysis
# ------------------------------------------------------------------------------

import os
import re
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

def plot_ae_training_history(path, model_type):
    """
    Plots training and validation loss history for a given model type.
    
    Args:
        path (str): Path to the directory containing the 'history.txt' file.
        model_type (str): Type of model ('mmae' or 'mmvae').
    """
    # Open and read the history file
    history_file = os.path.join(path, 'history.txt')
    with open(history_file, 'r') as fhistory:
        history = fhistory.readlines()

    # Initialize loss history array
    loss_hist = np.zeros((7 if model_type == 'mmae' else 9,))
    match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')

    # Extract loss values from history file
    for line in history:
        loss = np.array([float(x) for x in re.findall(match_number, line.strip())])
        loss_hist = np.vstack((loss_hist, loss))

    # Plot training and validation loss
    fig = plt.figure()
    for i, label in enumerate([
        'training (Radiation Parameters)', 'training (Outboard Plasma Profiles)',
        'training (Divertor Plasma Profiles)', 'validation (Radiation Parameters)',
        'validation (Outboard Plasma Profiles)', 'validation (Divertor Plasma Profiles)'
    ] + (['training (N(z))', 'validation (N(z))'] if model_type == 'mmvae' else [])):
        plt.plot(loss_hist[1:, 0], loss_hist[1:, i + 1], label=label)

    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (a.u.)')
    plt.legend()
    plt.grid()

    # Save the plot
    outfile = os.path.join(path, f'{model_type}_history.png')
    print(f'Saving training history as ({outfile})')
    plt.savefig(outfile)

# ------------------------------------------------------------------------------
def plot_ae_validation_example(path, k):
    """
    Plots a validation example comparing true and predicted values.

    Args:
        path (str): Path to the directory containing the 'validation.npz' file.
        k (int): Index of the validation example to plot.
    """
    # Load validation data
    data = np.load(os.path.join(path, 'validation.npz'))
    test_scal = data['test_scal']
    test_prf1 = data['test_prf1']
    test_prf2 = data['test_prf2']
    pred_scal = data['pred_scal']
    pred_prf1 = data['pred_prf1']
    pred_prf2 = data['pred_prf2']

    # Check index validity
    if k < 0 or k >= test_scal.shape[0]:
        print('Invalid index')
        return

    # Create subplots for true and predicted data
    fig, ax = plt.subplots(2, 5, figsize=(20, 8))
    
    # Plot true data
    ax[0, 0].plot(test_scal[k, :], label='Rad info')
    ax[0, 1].plot(test_prf1[k, :, 0], label='Te,u')
    ax[0, 1].plot(test_prf1[k, :, 1], label='Ne,u')
    ax[0, 2].plot(test_prf2[k, :, 0], label='Qt,in')
    ax[0, 2].plot(test_prf2[k, :, 1], label='Qt,out')
    ax[0, 3].plot(test_prf2[k, :, 2], label='Jsat,in')
    ax[0, 3].plot(test_prf2[k, :, 3], label='Jsat,out')
    ax[0, 4].plot(test_prf2[k, :, 4], label='Te,in')
    ax[0, 4].plot(test_prf2[k, :, 5], label='Te,out')

    # Plot predicted data
    ax[1, 0].plot(pred_scal[k, :], label='Rad info')
    ax[1, 1].plot(pred_prf1[k, :, 0], label='Te,u')
    ax[1, 1].plot(pred_prf1[k, :, 1], label='Ne,u')
    ax[1, 2].plot(pred_prf2[k, :, 0], label='Qt,in')
    ax[1, 2].plot(pred_prf2[k, :, 1], label='Qt,out')
    ax[1, 3].plot(pred_prf2[k, :, 2], label='Jsat,in')
    ax[1, 3].plot(pred_prf2[k, :, 3], label='Jsat.out')
    ax[1, 4].plot(pred_prf2[k, :, 4], label='Te,in')
    ax[1, 4].plot(pred_prf2[k, :, 5], label='Te,out')

    # Save the plot
    outfile = os.path.join(path, 'validation_example.png')
    print(f'Saving validation example as ({outfile})')
    plt.savefig(outfile)

# ------------------------------------------------------------------------------
def plot_ae_validation_statistics(path):
    """
    Plots validation statistics including absolute error and error distribution.

    Args:
        path (str): Path to the directory containing the 'validation.npz' file.
    """
    # Load validation data
    data = np.load(os.path.join(path, 'validation.npz'))
    test_scal = data['test_scal']
    test_prf1 = data['test_prf1']
    test_prf2 = data['test_prf2']
    pred_scal = np.squeeze(data['pred_scal'])
    pred_prf1 = np.squeeze(data['pred_prf1'])
    pred_prf2 = np.squeeze(data['pred_prf2'])

    # Calculate errors
    scal_error = pred_scal - test_scal
    prf1_error = pred_prf1 - test_prf1
    prf2_error = pred_prf2 - test_prf2

    test_prf = np.concatenate((test_prf1, test_prf2), axis=-1)
    prf_error = np.concatenate((prf1_error, prf2_error), axis=-1)

    # Plot absolute error vs true values
    fig, axs = plt.subplots(3, 3, figsize=(12, 6))
    plt.subplots_adjust(hspace=0.75)
    fig.suptitle("Validation Absolute Error vs True Value", fontsize=18, y=0.95)

    for i, ax in zip(range(9), axs.ravel()):
        if i == 0:
            ax.scatter(test_scal.flatten(), scal_error.flatten(), s=1)
        else:
            ax.scatter(test_prf[:, :, i-1].flatten(), prf_error[:, :, i-1].flatten(), s=1)
        ax.set_xlabel('True')
        ax.set_ylabel('Absolute Error')

    # Save the plot
    outfile = os.path.join(path, 'validation_abs_error.png')
    print(f'Saving validation statistics as ({outfile})')
    plt.savefig(outfile)

# ------------------------------------------------------------------------------
def plot_lsr_distribution(path):
    """
    Plots the distribution of latent space representations (LSR).

    Args:
        path (str): Path to the directory containing the 'z.npz' file.
    """
    with np.load(os.path.join(path, 'z.npz')) as data:
        z = data['z']

    nsamples, zdims = z.shape
    nbins = round(nsamples / 100)
    rows = int(np.ceil(zdims / 4))

    fig, axs = plt.subplots(nrows=rows, ncols=4, figsize=(12, 2 * rows))
    plt.subplots_adjust(hspace=0.75)
    fig.suptitle("LSR Distribution", fontsize=18, y=0.95)

    for i, ax in zip(range(zdims), axs.ravel()):
        ax.hist(z[:, i], bins=nbins)
        ax.set_title(f'LS Variable {i + 1}')

    outfile = os.path.join(path, 'lsr_distribution.png')
    print(f'Saving LSR distribution as ({outfile})')
    plt.savefig(outfile)

# ------------------------------------------------------------------------------
def plot_mlp_training_history(path, history):
    """
    Plots the training and validation loss for an MLP model.

    Args:
        path (str): Path to save the plot.
        history (object): Training history object containing loss and val_loss.
    """
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss Function')
    plt.plot(history.epoch, np.array(history.history['loss']), label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']), label='Val Loss')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()

    outfile = os.path.join(path, 'mlp_history.png')
    print(f'Saving MLP training history as ({outfile})')
    plt.savefig(outfile)

# ------------------------------------------------------------------------------
def plot_mlp_validation_statistics(path):
    """
    Plots validation statistics for an MLP model, including true vs predicted values,
    absolute error, and error distributions.

    Args:
        path (str): Path to the directory containing the 'mlp_validation.npz' file.
    """
    with np.load(os.path.join(path, 'mlp_validation.npz')) as data:
        z_true = data['z_true']
        z_pred = data['z_pred']

    z_abs_error = z_pred - z_true
    z_rel_error = z_abs_error / z_true
    nsamples, zdims = z_true.shape
    rows = int(np.ceil(zdims / 4))

    # True vs Predicted
    fig, axs = plt.subplots(nrows=rows, ncols=4, figsize=(12, 2 * rows))
    plt.subplots_adjust(hspace=0.75)
    fig.suptitle("MLP Performance: True vs Predicted", fontsize=18, y=0.95)

    for i, ax in zip(range(zdims), axs.ravel()):
        ax.scatter(z_true[:, i], z_pred[:, i], s=1)
        ax.plot([-10, 10], [-10, 10], 'k')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(f'LS Variable {i + 1}')

    outfile = os.path.join(path, 'mlp_performance.png')
    print(f'Saving MLP performance as ({outfile})')
    plt.savefig(outfile)

# ------------------------------------------------------------------------------
def plot_combined_validation_example(path, k):
    """
    Plots a validation example for the combined model.

    Args:
        path (str): Path to the directory containing the 'combined_validation.npz' file.
        k (int): Index of the validation example to plot.
    """
    with np.load(os.path.join(path, 'combined_validation.npz')) as data:
        rads = data['rads']
        prf = np.stack((
            data['teu'], data['neu'], data['qtl'] / 1e6, data['qtr'] / 1e6,
            data['jl'] / 1e6, data['jr'] / 1e6, data['tel'], data['ter']), axis=-1)
        y_rads = np.squeeze(data['y_rads'])
        y_prf = np.stack((
            data['y_teu'], data['y_neu'], data['y_qtl'] / 1e6, data['y_qtr'] / 1e6,
            data['y_jl'] / 1e6, data['y_jr'] / 1e6, data['y_tel'], data['y_ter']), axis=-1)

    if k < 0 or k >= rads.shape[0]:
        print('Invalid index')
        return

    fig, ax = plt.subplots(2, 5, figsize=(20, 8))

    # True values
    ax[0, 0].plot(rads[k, :], label='Rads')
    ax[0, 1].plot(prf[k, :, 0], label='Te,u')
    ax[0, 1].plot(prf[k, :, 1], label='Ne,u')
    ax[0, 2].plot(prf[k, :, 2], label='Qt,in')
    ax[0, 2].plot(prf[k, :, 3], label='Qt,out')
    ax[0, 3].plot(prf[k, :, 4], label='Jsat,in')
    ax[0, 3].plot(prf[k, :, 5], label='Jsat,out')
    ax[0, 4].plot(prf[k, :, 6], label='Te,in')
    ax[0, 4].plot(prf[k, :, 7], label='Te,out')
    
    # Predicted values
    ax[1, 0].plot(y_rads[k, :], label='Rads (pred)')
    ax[1, 1].plot(y_prf[k, :, 0], label='Te,u (pred)')
    ax[1, 1].plot(y_prf[k, :, 1], label='Ne,u (pred)')
    ax[1, 2].plot(y_prf[k, :, 2], label='Qt,in (pred)')
    ax[1, 2].plot(y_prf[k, :, 3], label='Qt,out (pred)')
    ax[1, 3].plot(y_prf[k, :, 4], label='Jsat,in (pred)')
    ax[1, 3].plot(y_prf[k, :, 5], label='Jsat,out (pred)')
    ax[1, 4].plot(y_prf[k, :, 6], label='Te,in (pred)')
    ax[1, 4].plot(y_prf[k, :, 7], label='Te,out (pred)')

    outfile = os.path.join(path, 'combined_validation_example.png')
    print(f'Saving combined validation example as ({outfile})')
    plt.savefig(outfile)

# ------------------------------------------------------------------------------
def plot_combined_validation_statistics(path):
    """
    Plots validation statistics (mean absolute error, standard deviation, R2)
    for the combined model.

    Args:
        path (str): Path to the directory containing the 'combined_validation.npz' file.
    """
    with np.load(os.path.join(path, 'combined_validation.npz')) as data:
        rads = data['rads']
        prf = np.stack((
            data['teu'], data['neu'], data['qtl'] / 1e6, data['qtr'] / 1e6,
            data['jl'] / 1e6, data['jr'] / 1e6, data['tel'], data['ter']), axis=-1)
        y_rads = np.squeeze(data['y_rads'])
        y_prf = np.stack((
            data['y_teu'], data['y_neu'], data['y_qtl'] / 1e6, data['y_qtr'] / 1e6,
            data['y_jl'] / 1e6, data['y_jr'] / 1e6, data['y_tel'], data['y_ter']), axis=-1)

    rad_abs_error = y_rads - rads
    prf_abs_error = y_prf - prf

    # Mean absolute error
    prf_ae_mean = np.mean(prf_abs_error, axis=0)

    fig, axs = plt.subplots(3, 3, figsize=(12, 6))
    plt.subplots_adjust(hspace=0.75)
    fig.suptitle("Combined Model Validation (Mean Absolute Error)", fontsize=18, y=0.95)

    for i, ax in zip(range(9), axs.ravel()):
        if i > 0:
            ax.plot(prf_ae_mean[:, i - 1])
            ax.set_title(f'Feature {i}')
            ax.set_ylabel('Mean Abs Error')

    outfile = os.path.join(path, 'combined_validation_mae.png')
    print(f'Saving combined validation statistics as ({outfile})')
    plt.savefig(outfile)
