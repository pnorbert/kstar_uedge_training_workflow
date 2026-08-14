#!/usr/bin/env python3
"""
Train an MLP that maps UEDGE input parameters into the autoencoder latent space.

Original by B. Zhu (zhu12@llnl.gov), last updated 03/11/2025.
Rewritten to read from downloaded ADIOS2 data and to be called as a function.
"""

from datetime import datetime
from pathlib import Path

import numpy as np

from DivControlNN.src.data import lsr_standardize, read_data_inputs
from DivControlNN.src.diagnose import plot_mlp_training_history, plot_mlp_validation_statistics
from DivControlNN.src.keras_compat import keras, tf

# Training parameters
EPOCHS = 10
neurons = 32
layers = 4
dropout_rate = 0.2
train_split = 0.9
batch_size = 256
initial_learning_rate = 1e-2
nsample = 72000
standardize_z = False

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=50000, decay_rate=0.95, staircase=True
)


def build_mlp_model(input_dim: int, output_dim: int):
    """Build the MLP that maps normalized control parameters to latent variables."""
    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            keras.layers.Dense(8, activation=tf.nn.relu, name="input_pars"),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(neurons, activation=tf.nn.relu),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(neurons, activation=tf.nn.relu),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_dim, activation=None, name="z_mlp_pred"),
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss="mse", optimizer=optimizer)
    return model


def train_mlp(inpath: Path, autoencoder_path: Path, model_id: str) -> Path:
    """Train an MLP from the downloaded inputs and an autoencoder's latent space."""
    print("TensorFlow version:", tf.__version__)

    model_name = f"mlp_dc_n{neurons}_l{layers}_{model_id}"
    training_set = inpath / "training_set.bp"

    ip, ncore, pinj, fz, diff = read_data_inputs(str(training_set))

    # Normalize input parameters using the original DivControlNN scaling.
    ip /= 1000.0
    ncore /= 8.0
    pinj /= 10.0
    fz *= 10.0
    diff /= 2.5
    inputs = np.squeeze(np.stack((ip, ncore, pinj, fz, diff), axis=1))

    latent_space_file = autoencoder_path / "z.npz"
    with np.load(latent_space_file) as data:
        latent_space = data["z"]
        if "sample_ids" in data:
            sample_ids = data["sample_ids"]
        elif latent_space.shape[0] == inputs.shape[0]:
            sample_ids = np.arange(inputs.shape[0])
        else:
            raise ValueError(
                f"Cannot align {inputs.shape[0]} MLP inputs with {latent_space.shape[0]} latent-space rows; "
                f"{latent_space_file} does not contain sample_ids."
            )

        if standardize_z:
            latent_space, _, _ = lsr_standardize(latent_space)

    if sample_ids.ndim != 1 or sample_ids.shape[0] != latent_space.shape[0]:
        raise ValueError("Latent-space sample_ids must contain one input row index per latent-space row.")
    if sample_ids.size and (sample_ids.min() < 0 or sample_ids.max() >= inputs.shape[0]):
        raise ValueError("Latent-space sample_ids reference rows outside the MLP input data.")

    inputs = inputs[sample_ids]
    ndata = min(inputs.shape[0], latent_space.shape[0], nsample)
    if ndata < 2:
        raise ValueError(f"Need at least 2 aligned samples to train the MLP, found {ndata}.")
    inputs = inputs[:ndata]
    latent_space = latent_space[:ndata]

    print(f"Splitting the data ({train_split} for training)")
    train_ids = np.random.choice(np.arange(ndata), int(train_split * ndata), replace=False)
    test_ids = np.setdiff1d(np.arange(ndata), train_ids)

    train_data = inputs[train_ids]
    train_labels = latent_space[train_ids]
    test_data = inputs[test_ids]
    test_labels = latent_space[test_ids]

    model = build_mlp_model(train_data.shape[1], train_labels.shape[1])
    model.summary()

    outpath = Path("models") / model_name
    outpath.mkdir(parents=True, exist_ok=True)
    checkpoint_file = outpath / "best_val.weights.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_file,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
    )

    tstart = datetime.now().strftime("%H:%M:%S")
    print("MLP training starts at ", tstart)
    history = model.fit(
        train_data,
        train_labels,
        epochs=EPOCHS,
        validation_split=0.2,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[checkpoint],
    )
    plot_mlp_training_history(outpath, history)
    print("MLP training completes at ", datetime.now().strftime("%H:%M:%S"))

    model.load_weights(checkpoint_file)
    test_pred = model.predict(test_data)

    np.savez(outpath / "mlp_validation", z_true=test_labels, z_pred=test_pred)
    plot_mlp_validation_statistics(outpath)

    return outpath
