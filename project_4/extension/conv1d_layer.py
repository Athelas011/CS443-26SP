'''conv1d_layer.py
1D Convolutional layer for time-series data.
Ariel Pan, Daniel Lyu
CS 443: Bio-inspired Machine Learning
Project 4 Extension: CNN-GRU for Time Series Forecasting
'''
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import tensorflow as tf
import layers


class Conv1D(layers.Layer):
    '''1D convolutional layer operating on (B, T, C_in) inputs.

    Applies a learned 1D convolution across the time axis using SAME padding,
    so the output sequence length T is preserved. Inherits weight saving/loading
    and activation logic from the project-4 Layer base class.

    Architecture role (Sajjad et al. 2020): two stacked Conv1D layers extract
    local spatial/temporal patterns from the raw feature sequences before the
    GRU stages model long-range dependencies.
    '''

    def __init__(self, name, units, kernel_size=2, stride=1, activation='relu',
                 prev_layer_or_block=None, wt_init='he'):
        '''Conv1D constructor.

        Parameters:
        -----------
        name: str. Human-readable layer name.
        units: int. Number of convolutional filters (C_out).
        kernel_size: int. Length of each 1-D filter window.
        stride: int. Stride along the time axis.
        activation: str. Activation function ('relu', 'linear', etc.).
        prev_layer_or_block: Layer. Previous layer in the network chain.
        wt_init: str. Weight init strategy: 'he' or 'normal'.
        '''
        super().__init__(layer_name=name, activation=activation,
                         prev_layer_or_block=prev_layer_or_block)
        self.units = units
        self.kernel_size = kernel_size
        self.stride = stride
        self.wt_init = wt_init

    def has_wts(self):
        return self.wts is not None

    def init_params(self, input_shape):
        '''Lazy-initialize weights and bias.

        Parameters:
        -----------
        input_shape: tuple/list. Shape of input tensor: (B, T, C_in).
        '''
        C_in = int(input_shape[-1])
        if self.wt_init == 'he':
            fan_in = self.kernel_size * C_in
            gain = float(self.get_kaiming_gain())
            stddev = gain * float(tf.math.sqrt(1.0 / tf.cast(fan_in, tf.float32)))
        else:
            stddev = 1e-3
        # tf.nn.conv1d filters: (kernel_size, C_in, C_out)
        self.wts = tf.Variable(
            tf.random.normal((self.kernel_size, C_in, self.units), stddev=stddev),
            name=self.layer_name + '_wts', trainable=True
        )
        self.bias = tf.Variable(
            tf.zeros((self.units,)),
            name=self.layer_name + '_bias', trainable=True
        )

    def compute_net_input(self, x):
        '''Apply 1D convolution with SAME padding.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, T, C_in).

        Returns:
        --------
        tf.float32 tensor. shape=(B, T, C_out).
        '''
        if self.wts is None:
            self.init_params(x.shape)
        return tf.nn.conv1d(x, self.wts, stride=self.stride, padding='SAME') + self.bias

    def __str__(self):
        return f'Conv1D layer output({self.layer_name}) shape: {self.output_shape}'
