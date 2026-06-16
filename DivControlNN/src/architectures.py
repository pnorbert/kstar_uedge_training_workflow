import tensorflow as tf
from tensorflow import keras
from .losses import sampling

# ------------------------------------------------------------------------------ #
# Utility Function: Apply a list of layers to an input sequentially
# ------------------------------------------------------------------------------ #
def apply_layers_on(layers, x):
    """
    Apply a list of keras layers onto a given input sequentially.

    Args:
        layers (list): List of keras.layers.Layer objects.
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Output tensor after applying all layers.
    """
    assert isinstance(layers, list), "Layers must be provided as a list."
    assert all([isinstance(l, keras.layers.Layer) for l in layers]), \
        "All elements in the layers list must be keras Layer objects."
    
    for layer in layers:
        x = layer(x)
    return x

# ------------------------------------------------------------------------------ #
# Encoder and Decoder Definitions for `example2_lite`
# ------------------------------------------------------------------------------ #
def mm_scal_mlp(input_dim, latent_dim):
    """
    Define a simple MLP-based encoder and decoder for scalar input.

    Args:
        input_dim (tuple): Shape of the scalar input.
        latent_dim (int): Latent space dimensionality.

    Returns:
        tuple: Encoder and decoder layers for scalar input.
    """
    print(f"--- architectures.py:mm_scal_mlp() input_dim = {input_dim}, latent_dim = {latent_dim}")
    # Encoder definition
    encoder = [
        keras.layers.Input(shape=input_dim, name='input_scal'),
        keras.layers.Flatten(),
        keras.layers.Dense(8),  # Hidden layer with 8 units
        keras.layers.Dense(8),  # Hidden layer with 8 units
        keras.layers.Dense(latent_dim, activation=None, name='z_scal')  # Latent representation
    ]
    
    # Decoder definition
    decoder = [
        keras.layers.Input(shape=(latent_dim,), name='zdec_scal'),
        keras.layers.Dense(8),  # Hidden layer with 8 units
        keras.layers.Dense(8),  # Hidden layer with 8 units
        keras.layers.Dense(input_dim[0] * input_dim[1], activation=None),  # Reconstruct flattened input
        keras.layers.Reshape(input_dim, name='recons_scal')  # Reshape to original input dimensions
    ]
    
    return encoder, decoder

# ------------------------------------------------------------------------------
# for upstream electron temperature and density radial profile (24,2)
def mm_prf1_cnn(input_dim, latent_dim):
    """
    Define a CNN-based encoder and decoder for profile 1 input.

    Args:
        input_dim (tuple): Shape of the profile input.
        latent_dim (int): Latent space dimensionality.

    Returns:
        tuple: Encoder and decoder layers for profile 1 input.
    """
    # Encoder definition
    encoder = [keras.layers.Input(shape=input_dim, name='input_prf1'),
               keras.layers.Conv1D(8, kernel_size=4),   # (21,8)
               keras.layers.MaxPool1D(pool_size=2),     # (10,8)
               #keras.layers.BatchNormalization(),
               keras.layers.Conv1D(4, kernel_size=5),   # (6,4)
               keras.layers.BatchNormalization(),
               keras.layers.Flatten(),
               keras.layers.Dense(latent_dim, activation=None, name='z_prf1'),
               ]
    # Decoder definition
    k = (6, 4) # Intermediate shape for reshaping during decoding
    decoder = [keras.layers.Input(shape=(latent_dim,), name='zdec_prf1'),
               #keras.layers.BatchNormalization(),
               keras.layers.Dense(k[0] * k[1]),
               keras.layers.BatchNormalization(),
               keras.layers.Reshape(k),                 # (6,4)
               keras.layers.Conv1DTranspose(8, kernel_size=6),  # (11,8)
               keras.layers.UpSampling1D(size=2),               # (22,8)
               #keras.layers.BatchNormalization(),
               keras.layers.Conv1DTranspose(2, kernel_size=6),  # (27,2)
               keras.layers.Cropping1D((1, 2), name='recons_prf1'),  # (24,2)
               ]

    return encoder, decoder

# ------------------------------------------------------------------------------
# for electron temprature, current and heat flux on target plates (24,6)
def mm_prf2_cnn(input_dim, latent_dim):
    """
    Define a CNN-based encoder and decoder for profile 2 input.

    Args:
        input_dim (tuple): Shape of the profile input.
        latent_dim (int): Latent space dimensionality.

    Returns:
        tuple: Encoder and decoder layers for profile 2 input.
    """
    # Encoder definition
    encoder = [keras.layers.Input(shape=input_dim, name='input_prf2'),
               keras.layers.Conv1D(16, kernel_size=4),  # (21,16)
               keras.layers.MaxPool1D(pool_size=2),     # (10,16)
               #keras.layers.BatchNormalization(),
               keras.layers.Conv1D(8, kernel_size=5),   # (6,8)
               keras.layers.BatchNormalization(),
               keras.layers.Flatten(),
               keras.layers.Dense(latent_dim, activation=None, name='z_prf2'),
               ]
    
    # Decoder definition
    k = (6, 8) # Intermediate shape for reshaping during decoding
    decoder = [keras.layers.Input(shape=(latent_dim,), name='zdec_prf2'),
               #keras.layers.BatchNormalization(),
               keras.layers.Dense(k[0] * k[1]),
               keras.layers.BatchNormalization(),
               keras.layers.Reshape(k),                 # (6,8)
               keras.layers.Conv1DTranspose(16, kernel_size=6), # (11,16)
               keras.layers.UpSampling1D(size=2),               # (22,16)
               #keras.layers.BatchNormalization(),
               keras.layers.Conv1DTranspose(6, kernel_size=6),  # (27,16)
               keras.layers.Cropping1D((1, 2), name='recons_prf2'),  # (24,6)
               ]

    return encoder, decoder

# ------------------------------------------------------------------------------
def multimodal_dc(input_shp, latent_dim, feat_sz, do_vae=True):
    """
    Define a multimodal autoencoder for scalar and two profile inputs.

    Args:
        input_shp (tuple): Tuple containing input shapes for scalar, profile 1, and profile 2 inputs.
        latent_dim (int): Latent space dimensionality.
        feat_sz (tuple): Feature sizes for scalar, profile 1, and profile 2 inputs.
        do_vae (bool): Whether to use a Variational Autoencoder (VAE) architecture.

    Returns:
        tuple: Encoder and decoder Keras models.
    """
    print(f"--- architectures.py:multimodal_dc() input_shp = {input_shp}, latent_dim = {latent_dim}"
          f" feat_sz = {feat_sz} do_vae = {do_vae}"
         )

    # Validate input shapes
    assert isinstance(input_shp, tuple) and len(input_shp) == 3, \
        f'Invalid input shape: {input_shp}'
    dim_scal, dim_prf1, dim_prf2 = input_shp
    assert isinstance(dim_scal, tuple) and len(dim_scal) == 2, \
        f'Invalid scalar size: {dim_scal}'
    assert isinstance(dim_prf1, tuple) and len(dim_prf1) == 2, \
        f'Invalid profile size: {dim_prf1}'
    assert isinstance(dim_prf2, tuple) and len(dim_prf2) == 2, \
        f'Invalid profile size: {dim_prf2}'
    assert isinstance(feat_sz, tuple) and len(feat_sz) == 3, \
        f'Need a tuple of two values for feature sizes, found: {feat_sz}'

    feat_sz_scal, feat_sz_prf1, feat_sz_prf2 = int(feat_sz[0]), int(feat_sz[1]), int(feat_sz[2])

    # Define encoders and decoders for each modality
    encoder_scal, decoder_scal = mm_scal_mlp(dim_scal, feat_sz_scal)
    encoder_prf1, decoder_prf1 = mm_prf1_cnn(dim_prf1, feat_sz_prf1)
    encoder_prf2, decoder_prf2 = mm_prf2_cnn(dim_prf2, feat_sz_prf2)

    # Define the aggregator (latent space encoder and decoder)
    if do_vae:
        encoder_agg = [
            keras.layers.Dense(2 * latent_dim, activation='relu'),
            keras.layers.Dense(int(1.5 * latent_dim), activation='relu'),
            (keras.layers.Dense(latent_dim, name='z_mu'), keras.layers.Dense(latent_dim, name='z_logvar')),
            keras.layers.Lambda(sampling, name='z')  # Sampling layer for VAE
        ]
    else:
        encoder_agg = [
            keras.layers.Dense(2 * latent_dim, activation='relu'),
            keras.layers.Dense(int(1.5 * latent_dim), activation='relu'),
            keras.layers.Dense(latent_dim, activation=None, name='z')
        ]
    
    decoder_agg = [
        keras.layers.Dense(int(1.5 * latent_dim), activation='relu'),
        keras.layers.Dense(2 * latent_dim, activation='relu'),
        keras.layers.Dense(feat_sz_scal + feat_sz_prf1 + feat_sz_prf2, activation='relu')
    ]

    # Build the encoder model
    input_scal, z_scal = encoder_scal[0], apply_layers_on(encoder_scal[1:], encoder_scal[0])
    input_prf1, z_prf1 = encoder_prf1[0], apply_layers_on(encoder_prf1[1:], encoder_prf1[0])
    input_prf2, z_prf2 = encoder_prf2[0], apply_layers_on(encoder_prf2[1:], encoder_prf2[0])
    
    # Concatenate latent features from all modalities
    z_mm = keras.layers.concatenate([z_scal, z_prf1, z_prf2], name='multimodal_concatenation')
    
    if do_vae:
        z_mm = apply_layers_on(encoder_agg[:2], z_mm)
        z_mu = encoder_agg[2][0](z_mm)
        z_log_var = encoder_agg[2][1](z_mm)
        z_mm = encoder_agg[3]([z_mu, z_log_var])
    else:
        z_mm = apply_layers_on(encoder_agg, z_mm)
    
    encoder = keras.models.Model(inputs=[input_scal, input_prf1, input_prf2], outputs=z_mm, name='encoder_multimodal')

    # Build the decoder model
    input_z = keras.layers.Input(shape=(latent_dim,), name='input_dec')
    z_mm = apply_layers_on(decoder_agg, input_z)
    z_scal, z_prf1, z_prf2 = tf.split(z_mm, [feat_sz_scal, feat_sz_prf1, feat_sz_prf2], axis=1)

    # Reconstruct each modality
    z_scal = apply_layers_on(decoder_scal[1:], z_scal)
    z_prf1 = apply_layers_on(decoder_prf1[1:], z_prf1)
    z_prf2 = apply_layers_on(decoder_prf2[1:], z_prf2)

    decoder = keras.models.Model(inputs=input_z, outputs=[z_scal, z_prf1, z_prf2], name='decoder_multimodal')

    return encoder, decoder

# ------------------------------------------------------------------------------
