import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from collections import defaultdict

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config

from . import architectures

from . import losses
from .losses import sampling

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class Autoencoder(keras.Model):

    def __init__(self, arch_name, input_shp, feature_sz, latent_dim, do_vae=False):
        super(Autoencoder, self).__init__()

        self.arch_name = arch_name
        self.input_shp = input_shp
        # we will create features for images, profile and signal separately
        # if applicable and then combine them into a latent space
        self.feat_sz   = feature_sz
        self.z_dim = latent_dim
        self.is_vae = do_vae
        self.num_components = len(input_shp)

        #LOGGER.info(f'Setting up Autoencoder: (variational={self.is_vae}) '
        print(f'Setting up Autoencoder: (variational={self.is_vae}) '
              f'{self.arch_name} : {self.input_shp} --> {self.z_dim}')

        arch = getattr(architectures, self.arch_name)
        self.encoder, self.decoder = arch(self.input_shp, self.z_dim, self.feat_sz, self.is_vae)


        '''
        # ----------------------------------------------------------------------
        # set up the model
        # ----------------------------------------------------------------------
        # encoder
        nlayers_encoder = len(encoder)
        print ('nlayers_encoder:', nlayers_encoder)

        # if this is a vae, then we skip the last layer
        # and instead add the vae layer (mu, sigma) --> z
        if self.is_vae:
            nlayers_encoder -= 1

        x = encoder[0]
        for i, l in enumerate(encoder[1:nlayers_encoder]):
            x = l(x)
        z = x  # output of the model is the latent space

        # this is the part that makes this a variational AE!
        if self.is_vae:
            z_mu = keras.layers.Dense(units=latent_dim, name="z_mu")(z)
            z_log_var = keras.layers.Dense(units=latent_dim, name="z_logvar")(z)

            # now, this is the latent space
            z = keras.layers.Lambda(sampling, name="z")([z_mu, z_log_var])

        # decoder
        x = encoder[0]  # need to store x
        y = z
        for i, l in enumerate(decoder[1:]):
            y = l(y)

        assert x.shape.as_list() == y.shape.as_list(), \
                                            f'Invalid architecture: ' \
                                            f'x = {x.shape}, y = {y.shape}'
        assert z.shape.as_list() == decoder[0].shape.as_list(), \
                                            f'Invalid architecture: ' \
                                            f'z = {z.shape}, ' \
                                            f'decoder_input = {decoder[0].shape}'

        # ----------------------------------------------------------------------
        # turn this model into keras model
        # ----------------------------------------------------------------------
        self.encoder = keras.models.Model(x, z, name="encoder")
        self.decoder = keras.models.Model(z, y, name="decoder")
        #self.encoder = keras.Model(x, z, name="encoder")
        #self.decoder = keras.Model(z, y, name="decoder")
        #self.encoder = keras.Model(x, z)
        #self.decoder = keras.Model(z, y)
        '''

        if True:
            self.encoder.summary(line_length=140)
            self.decoder.summary(line_length=140)

        # ----------------------------------------------------------------------
        self.optimizer = None

        # define weights
        '''
        self.custom_weights = np.ones(dims[0], dtype=np.float32)
        self.custom_weights[5:] = 0.02
        LOGGER.info(f'Defined custom weights: {self.custom_weights.shape}; '
                    f'[{self.custom_weights.min()}, {self.custom_weights.max()}]')
        '''

    # --------------------------------------------------------------------------
    def encode(self, x):
        return self.encoder(x).numpy()

    def decode(self, z):
        return self.decoder(z).numpy()

    def reconstruct(self, x):
        return self.decoder(self.encoder(x))

    def write_model_summary(self, outpath):
        with open(os.path.join(outpath, 'model_encoder_summary.txt'), 'w') as f:
            self.encoder.summary(print_fn=lambda x: f.write(x + '\n'))
        with open(os.path.join(outpath, 'model_decoder_summary.txt'), 'w') as f:
            self.decoder.summary(print_fn=lambda x: f.write(x + '\n'))

    # --------------------------------------------------------------------------
    # basic train test functions
    # --------------------------------------------------------------------------
    @tf.function
    def train_step(self, data):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            total_loss, loss_metrics = self.compute_losses(data, training=True)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss_metrics

    @tf.function
    def test_step(self, data):
        total_loss, loss_metrics = self.compute_losses(data, training=False)
        return loss_metrics

    @tf.function
    def compute_losses(self, x, training=True):
        np_config.enable_numpy_behavior()

        z = self.encoder(x, training=training)
        y = self.decoder(z, training=training)

        if self.num_components == 3:
            xscal, xprf1, xprf2 = x[0], x[1], x[2]
            yscal, yprf1, yprf2 = y[0], y[1], y[2]
        else:
            raise ValueError("Unsupported number of components. Customization required.")

        if len(xscal.shape) == 2: # for scalars, add addtional dimension
            xscal = xscal.reshape(xscal.shape+(1,))

        # Compute losses for scalar inputs
        # loss_mse = tf.keras.losses.MeanSquaredError()(x, y)
        loss_mse_scal = losses.weighted_loss_mse(xscal, yscal, 2)
        loss_dict = {'mse_scal': loss_mse_scal}
        
        # Compute losses for profile inputs
        loss_mse_prf1 = losses.weighted_loss_mse(xprf1, yprf1, 2)
        loss_dict['mse_prfa'] = loss_mse_prf1
        loss_mse_prf2 = losses.weighted_loss_mse(xprf2, yprf2, 6)
        loss_dict['mse_prfb'] = loss_mse_prf2

        if not self.is_vae:
            return loss_mse_scal + loss_mse_prf1 + loss_mse_prf2, loss_dict

        # Add VAE loss if applicable
        loss_vae = 1.e-7*losses.vae_loss(z, z)
        loss_dict['vae'] = loss_vae

        return loss_mse_scal + loss_mse_prf1 + loss_mse_prf2 + loss_vae, loss_dict

    # --------------------------------------------------------------------------
    # the main training loop
    # --------------------------------------------------------------------------
    def train(self, train_data, validation_data, epochs, batch_size, \
              learning_rate, outpath):

        # todo: this should come from config or function parameters
        # epochs = int(self.config['training']['nepochs'])
        # batch_size = int(self.config['training']['batch_size'])
        # epochs = 5
        # batch_size = 100
        # learning_rate = 0.1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        #self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        # ----------------------------------------------------------------------
        metrics = {}

        # ----------------------------------------------------------------------
        # input assertions
        assert isinstance(train_data, tuple)
        assert isinstance(validation_data, tuple)
        assert len(train_data) == len(validation_data)

        # temporary change until we add multiple modalities
        #train_data = (train_data[0])
        #validation_data = (validation_data[0])

        ncomps = len(train_data)
        ntrain = train_data[0].shape[0]
        nvalid = validation_data[0].shape[0]

        assert self.num_components == ncomps
        for i in range(1, ncomps):
            assert ntrain == train_data[i].shape[0]
            assert nvalid == validation_data[i].shape[0]

        train_data = (tf.data.Dataset.from_tensor_slices(train_data).shuffle(ntrain).batch(batch_size))
        validation_data = (tf.data.Dataset.from_tensor_slices(validation_data).shuffle(nvalid).batch(batch_size))

        fhistory = open(os.path.join(outpath,'history.txt'),'a')
        # ----------------------------------------------------------------------
        # for each epoch
        for epoch in range(1, epochs + 1):

            print (f'> Starting epoch {epoch}...', end='')

            metrics = {}

            # training for each batch
            training_metrics_per_batch = defaultdict(list)
            for x in train_data:
                m1 = self.train_step(x)
                for key, val in m1.items():
                    training_metrics_per_batch[key].append(val.numpy())


            # validation for each batch
            validation_metrics_per_batch = defaultdict(list)
            for x in validation_data:
                m2 = self.test_step(x)
                for key, val in m2.items():
                    validation_metrics_per_batch['val_'+key].append(val.numpy())

            # ------------------------------------------------------------------
            # collect the metrics from training and validation
            for key, val in training_metrics_per_batch.items():
                #metrics[key] = np.sum(val, axis=0)
                metrics[key] = np.mean(val, axis=0)

            for key, val in validation_metrics_per_batch.items():
                #metrics[key] = np.sum(val, axis=0)
                metrics[key] = np.mean(val, axis=0)

            print (f' done!', metrics)

            fhistory.write(str(epoch)+'\t'+str(metrics)+'\n')
            # ------------------------------------------------------------------

        # return all the metrics captured during training
        return metrics
   
        fhistory.close()

# ------------------------------------------------------------------------------
