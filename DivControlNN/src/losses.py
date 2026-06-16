
import tensorflow as tf
from tensorflow import keras


# ------------------------------------------------------------------------------
# sampling trick for vae
def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = keras.backend.random_normal(shape=keras.backend.shape(mu),
                                          mean=0.0, stddev=1.0)
    random_sample = mu + keras.backend.exp(log_variance/2) * epsilon
    return random_sample

# ------------------------------------------------------------------------------
# sampling trick for vae
def vae_loss(z_mean, z_log_var):
    """Calculate vae_loss = KL loss for each data in minibatch
    """
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    return kl_loss

# ------------------------------------------------------------------------------
def weighted_loss_mse(x_true, x_pred, weights = None):

    # calculating squared difference between target and predicted values
    loss = keras.backend.square(x_pred - x_true)  # (batch_size, x, y, c)

    # multiplying the values with weights along batch dimension
    if weights is not None:
        loss = loss * weights  # (batch_size, x, y, c)

    # summing both loss values along batch dimension
    # loss = keras.backend.mean(loss, axis=(1,2,3))  # (batch_size,)
    #loss = keras.backend.mean(loss)     # mean value for the entire batch
    loss = keras.backend.sqrt(keras.backend.mean(loss))     # sqrt of mean value for the entire batch
    return loss

# ------------------------------------------------------------------------------
def weighted_loss_mae(x_true, x_pred, weights = None):

    # calculating squared difference between target and predicted values
    loss = keras.backend.abs(x_pred - x_true)  # (batch_size, x, y, c)

    # multiplying the values with weights along batch dimension
    if weights is not None:
        loss = loss * weights  # (batch_size, x, y, c)

    # summing both loss values along batch dimension
    # loss = keras.backend.mean(loss, axis=(1,2,3))  # (batch_size,)
    loss = keras.backend.mean(loss)     # mean value for the entire batch
    return loss

# ------------------------------------------------------------------------------
