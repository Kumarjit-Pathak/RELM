# 3 stage VAE feature generator
from keras.layers import Dense, Input, Lambda
# from keras.models import Model,Sequential
import keras.models as krm
from keras.optimizers import Adam
from keras.layers.merge import Concatenate
from keras import backend as K
from keras import metrics

from .utils import plot_pair
import numpy as np


# todo: all things will be under class Latent Space
class LatentSpace(object):

    def __init__(self):
        pass

    def __get_continuous_latents(self):
        pass

    def __get_categorical_latents(self):
        pass

    def fit(self):
        pass

    def transform(self):
        pass

    def plot_features(self):
        pass

    def get_generators(self):
        pass

    def __vae_feature_generator(self):
        pass

    def __rbm_feature_generator(self):
        pass


def vae_feature_generation_3stg(data, components='auto', funnel=(45, 30, 10),
                                batch=200, epoch=250, epsilon=1.0, opt=None,
                                validation=False, plot_it=False):
    original_dim = data.shape[1]

    if components == 'auto':
        intermediate_dim_1 = 45
        intermediate_dim_2 = 30
        latent_dim = 10
    elif components == 'ratio':
        intermediate_dim_1 = int(np.ceil(original_dim / funnel[0]))
        intermediate_dim_2 = int(np.ceil(original_dim / funnel[1]))
        latent_dim = int(np.ceil(intermediate_dim_2 / funnel[2]))
    elif components == 'values':
        intermediate_dim_1 = funnel[0]
        intermediate_dim_2 = funnel[1]
        latent_dim = funnel[2]

    batch_size = batch
    epochs = epoch
    epsilon_std = epsilon

    x = Input(shape=(original_dim,))
    h_1 = Dense(intermediate_dim_1, activation='elu')(x)
    h_2 = Dense(intermediate_dim_2, activation='elu')(h_1)
    z_mean = Dense(latent_dim)(h_2)
    z_log_var = Dense(latent_dim)(h_2)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h_2 = Dense(intermediate_dim_2, activation='elu')
    decoder_h_1 = Dense(intermediate_dim_1, activation='elu')
    decoder_mean = Dense(original_dim, activation='relu')
    h_decoded_2 = decoder_h_2(z)
    h_decoded_1 = decoder_h_1(h_decoded_2)
    x_decoded_mean = decoder_mean(h_decoded_1)

    # instantiate VAE model
    vae = krm.Model(inputs=x, outputs=x_decoded_mean)

    # Compute VAE loss
    xent_loss = original_dim * metrics.mean_squared_error(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)
    if opt is None:
        opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    vae.compile(optimizer=opt)
    vae.summary()

    # fitting our generated features on the data
    vae.fit(data,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size
            #         validation_data=(x_test, None)
            )

    # Encoder model for feature generation
    encoder = krm.Model(inputs=x, outputs=z_mean)

    # display a 2D plot of the digit classes in the latent space
    tr = encoder.predict(data, batch_size=batch_size)
    if plot_it:
        print('\nPlotting features!\n')
        # Plotting with SNS
        plot_pair(tr)

    return tr, latent_dim, encoder


# todo: add bernoulli rbm here...
