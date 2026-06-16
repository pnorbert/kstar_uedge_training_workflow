# ------------------------------------------------------------------------------
# Utility Functions for Reading Training Data
# ------------------------------------------------------------------------------

import os
import glob
from adios2 import FileReader
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
def read_data_inputs(filename: str):
    """
    Reads input data from an ADIOS2 file.

    Args:
        filename (str): Path of the ADIOS2 file.

    Returns:
        tuple: Arrays for ip, ncore, pinj, fz, and diff.
    """
    with FileReader(filename) as f:
        vip = f.inquire_variable("ip")
        nsteps = vip.steps()
        ip = f.read("ip", step_selection=[0, nsteps])
        ncore = f.read("ncore", step_selection=[0, nsteps])
        pinj = f.read("pinj", step_selection=[0, nsteps])
        fz = f.read("fz", step_selection=[0, nsteps])
        diff = f.read("diff", step_selection=[0, nsteps])

    return (ip.astype(np.float32), ncore.astype(np.float32),
            pinj.astype(np.float32), fz.astype(np.float32),
            diff.astype(np.float32))

# ------------------------------------------------------------------------------
def read_data_outputs(filename: str):
    """
    Reads a lightweight version of example dataset 2 from an HDF5 file.

    Args
        filename (str): Path of the ADIOS2 file.

    Returns:
        tuple: Arrays for qtl, qtr, jl, jr, tel, ter, teu, neu, and rads.
    """

    with FileReader(filename) as f:
        vip = f.inquire_variable("ip")
        nsteps = vip.steps()
        neu = f.read("neu", step_selection=[0, nsteps])
        teu = f.read("teu", step_selection=[0, nsteps])
        ter = f.read("ter", step_selection=[0, nsteps])
        tel = f.read("tel", step_selection=[0, nsteps])
        jr = f.read("jr", step_selection=[0, nsteps])
        jl = f.read("jl", step_selection=[0, nsteps])
        qtr = f.read("qtr", step_selection=[0, nsteps])
        qtl = f.read("qtl", step_selection=[0, nsteps])
        rads = f.read("rads", step_selection=[0, nsteps])
        pinj = f.read("pinj", step_selection=[0, nsteps])

    # Normalize radiation data
    rads[:, 1] = rads[:, 1] / rads[:, 0]  # Divertor radiation fraction
    rads[:, 0] = rads[:, 0] / (pinj * 1e6)  # Total radiation fraction

    return (qtl.astype(np.float32), qtr.astype(np.float32),
            jl.astype(np.float32), jr.astype(np.float32),
            tel.astype(np.float32), ter.astype(np.float32),
            teu.astype(np.float32), neu.astype(np.float32),
            rads.astype(np.float32))

# ------------------------------------------------------------------------------
def lsr_standardize(data):
    """
    Standardizes the data by subtracting the mean and dividing by the standard deviation.
    Args:
        data: Input data to be standardized.
    Returns:
        Standardized data, mean, and standard deviation.
    """
    mean = np.mean(data, axis=-2, keepdims=True)
    std = np.std(data, axis=-2, keepdims=True)
    return (data - mean) / std, mean, std

# ------------------------------------------------------------------------------
def lsr_destandardize(data, mean, std):
    """
    De-standardizes the data by reversing the standardization process.
    Args:
        data: Standardized data.
        mean: Mean used during standardization.
        std: Standard deviation used during standardization.
    Returns:
        De-standardized data.
    """
    return data * std + mean

# ------------------------------------------------------------------------------
def standardize(**kwargs):
    """
    Standardizes the input data by subtracting the mean and dividing by the standard deviation.

    Args:
        kwargs (dict): Named arrays to standardize.

    Returns:
        list: Standardized arrays and their respective (mean, std) tuples.
    """
    stds = {}
    for k, v in kwargs.items():
        m, s = v.mean(), v.std()
        stds[k] = (m, s)
        print(f'Standardizing ({k}): {v.shape} : mean = {m}, std = {s}')
        v -= m
        v /= s
        print(f'Result: mean = {v.mean()}, std = {v.std()}')

    return [(v, stds[k]) for k, v in kwargs.items()]

# ------------------------------------------------------------------------------
def destandardize(**kwargs):
    """
    Reverts the standardization process.

    Args:
        kwargs (dict): Named arrays and their respective (mean, std) tuples.

    Returns:
        list: De-standardized arrays.
    """
    for k, (v, stds) in kwargs.items():
        m, s = stds
        v *= s
        v += m

    return [v for k, (v, stds) in kwargs.items()]

# ------------------------------------------------------------------------------
def maxmin_norm(v, mm=None):
    """
    Performs min-max normalization on the input data.

    Args:
        v (array): Input array to normalize.
        mm (tuple, optional): Min and max values for normalization.

    Returns:
        list: Normalized array and its (min, max) tuple.
    """
    if mm is None:
        vmin, vmax = np.floor(v.min()), np.ceil(v.max())
    else:
        vmin, vmax = mm

    v -= vmin
    v /= (vmax - vmin)

    return [v, (vmin, vmax)]

# ------------------------------------------------------------------------------
def maxmin_denorm(**kwargs):
    """
    Reverts the min-max normalization process.

    Args:
        kwargs (dict): Named arrays and their respective (min, max) tuples.

    Returns:
        list: De-normalized arrays.
    """
    for k, (v, mm) in kwargs.items():
        vmin, vmax = mm
        v *= (vmax - vmin)
        v += vmin

    return [v for k, (v, mm) in kwargs.items()]

