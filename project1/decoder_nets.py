'''decoder_net.py
Artificial neural networks that decoder signals encoding in the activations of the Hebbian network.
Ariel Pan, Daniel Lyu
CS 443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning
'''
import tensorflow as tf

import network
from layers import Dense

class LinearDecoder(network.DeepNetwork):
    '''Linear Decoder network with the following architecture:

    Dense output layer (softmax activation).

    Uses standard cross-entropy loss.
    '''
    def __init__(self, input_feats_shape, C):
        '''LinearDecoder contructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The FLATTEN shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32*32*3,).
        C: int.
            Number of classes in the dataset.

        TODO: Build the network and create instance variables as needed.

        NOTE:
        1. You should make use of the parent's constructor to initialize common variables.
        2. The output layer for ANY `DeepNetwork` here and going forward should be assigned to the variable
        self.output_layer.
        '''
        super().__init__(input_feats_shape)
        self.num_classes = C
        self.output_layer = Dense(name = 'linear_decoder_output_layer', units=C, activation='softmax', wt_scale=1e-3, prev_layer_or_block=None, wt_init='normal',
                 do_group_norm=False)
    def __call__(self, x):
        '''Do a forward pass thru the network with mini-batch `x`.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, M)
            The input mini-batch computed in the current layer.

        Returns:
        --------
        tf.float32 tensor. shape=(B, M).
            The output layer activation computed on the current mini-batch.
        '''
        return self.output_layer(x)

class NonlinearDecoder(network.DeepNetwork):
    '''Nonlinear Decoder network proposed by Krotov & Hopfield with the following architecture:

    Dense output layer (tanh activation).

    Uses lp loss.
    '''
    def __init__(self, input_feats_shape, C, wt_scale=0.1, beta=0.0025, loss_exp=2.0):
        '''LinearDecoder contructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The FLATTEN shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32*32*3,).
        C: int.
            Number of classes in the dataset.
        wt_scale: float.
            The standard deviation of the layer weights when initialized according to a standard normal distribution
            ('normal' method).
        beta: float.
            β hyperparameter inside the tanh activation function.
        loss_exp: float
            The exponent `m` in the lp loss function.

        TODO: Build the network and create instance variables as needed.

        NOTE:
        1. You should make use of the parent's constructor to initialize common variables.
        2. The output layer for ANY `DeepNetwork` here and going forward should be assigned to the variable
        self.output_layer.
        '''
        super().__init__(input_feats_shape)
        self.num_classes = C
        self.beta = beta
        self.loss_exp = loss_exp
        self.output_layer = Dense(name = 'nonlinear_decoder_output_layer', units=C, activation='tanh', wt_scale=wt_scale, prev_layer_or_block=None, wt_init='normal',
                 do_group_norm=False)
        self.output_layer.set_tanh_beta(beta)
        
    def __call__(self, x):
        '''Do a forward pass thru the network with mini-batch `x`.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, M)
            The input mini-batch computed in the current layer.

        Returns:
        --------
        tf.float32 tensor. shape=(B, M).
            The output layer activation computed on the current mini-batch.
        '''
        return self.output_layer(x)
