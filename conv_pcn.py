'''conv_pcn.py
Convolutional predictive coding neural network.
Daniel Lyu, Ariel Pan
CS 443: Bio-Inspired Learning
'''
import network
from conv_pcn_blocks import ConvPCNBlock
from layers import Dense, Flatten, Dropout
from conv_layers import Conv2D, MaxPool2D

class ConvPCN(network.DeepNetwork):
    '''Parent class for all specific ConvPCN network architectures.

    The general structure is:
    Input layer → Conv2D layer → ConvPCNBlock block → ... → ConvPCNBlock block → Flatten → Dropout → Dense → Dense

    Recall that each network uses softmax activation in the output layer and cross-entropy loss.
    '''
    def __init__(self, input_feats_shape):
        '''ConvPCN constructor

        Parameters:
        -----------
        input_feats_shape: tuple of ints.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        '''
        super().__init__(input_feats_shape=input_feats_shape)

    def __call__(self, x):
        '''Forward pass through the network

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, Iy, Ix, n_chans).
            Mini-batch of image data.

        Returns:
        -----------
        tf.float32 tensor. shape=(B, Iy, Ix, C).
            Activations produced by output layer to mini-batch.

        Hint: Although there is feedback, it is totally enclosed within each ConvPCNBlock. This means
        layers and blocks are sequentially connected like in a conventional neural network, which is convenient ;)
        '''
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class ConvPCN6Mini(ConvPCN):
    '''ConvPCN with 6 weight layers/blocks:

    - 1 solo `Conv2D` layer at the start of the network
    - 3 ConvPCNBlocks
    - 1 dense hidden layer
    - 1 output layer

    Here is the specific connectivity:
    Input layer → Conv2D layer → [ConvPCNBlock block]x3 → Flatten → Dropout → Dense hidden → Dense output
    '''
    def __init__(self, input_feats_shape, C, conv_units=64, pcn_units=(64, 128, 192), dense_units=128, dropout_rate=0.2,
                 maxpool_after_pcn_block=(True, True, True), num_steps=5, step_lr=0.5, wt_init='normal',
                 do_group_norm=False):
        '''ConvPCN6Mini constructor. Builds and configures the PCN, from input to output layers.

        Parameters:
        -----------
        input_feats_shape: tuple of ints.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        C: int.
            Number of classes in the dataset.
        conv_units: int.
            Number of filters the first 2D convolutional layer has in it.
        pcn_units: tuple of int.
            The number of units/filters in each PCN block. The length of the tuple specifies the number of PCNBlocks
            that are used. For example, (64, 128, 192) means there are 3 separate PCNBlocks with the respective number
            of filters therein.
        dense_units: int.
            Number of hidden units in the dense hidden layer of the net (H).
        dropout_rate: float or None.
            The dropout rate used in each `Dropout` layer in the net.
            If `None` passed in, then we do not use dropout in the net.
        maxpool_after_pcn_block: tuple of bool.
            Whether to follow each PCNBlock with a max pooling layer to downscale the spatial resolution. Length must
            match the length of `pcn_units`. If there is a max pooling layer, the stride and window size should both
            be set to 2.
        num_steps: int.
            Number of steps to use to update the block's state during each forward pass thru the block.
        state_lr: float.
            The learning rate used to iteratively update the block's state during each forward pass thru the block.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        do_group_norm. bool:
            Whether to do group normalization in each layer and block of the net.

        Reminders:
        - Add your layers to the layers list.
        - Make sure your output layer is assigned to self.output_layer.
        '''
        super().__init__(input_feats_shape)
  
        self.layers = []

        prev = None

        # --- Initial Conv Layer ---
        conv = Conv2D(
            name='Conv2D_init',
            units=conv_units,
            kernel_size=3,
            strides=1,
            activation='relu',
            prev_layer_or_block=prev,
            wt_init=wt_init,
            do_group_norm=do_group_norm
        )
        self.layers.append(conv)
        prev = conv

        # --- PCN Blocks ---
        for i, units in enumerate(pcn_units):
            block = ConvPCNBlock(
                blockname=f'PCNBlock_{i}',
                units=units,
                kernel_size=3,
                strides=1,
                num_steps=num_steps,
                state_lr=step_lr,
                dropout_rate=None,  # dropout handled later
                wt_init=wt_init,
                do_group_norm=do_group_norm,
                prev_layer_or_block=prev
            )
            self.layers.append(block)
            prev = block

            # Optional MaxPool
            if maxpool_after_pcn_block[i]:
                pool = MaxPool2D(
                    name=f'MaxPool_{i}',
                    pool_size=2,
                    strides=2,
                    prev_layer_or_block=prev
                )
                self.layers.append(pool)
                prev = pool

        # --- Flatten ---
        flatten = Flatten(
            name='Flatten',
            prev_layer_or_block=prev
        )
        self.layers.append(flatten)
        prev = flatten

        # --- Dropout ---
        if dropout_rate is not None:
            drop = Dropout(
                name='Dropout',
                rate=dropout_rate,
                prev_layer_or_block=prev
            )
            self.layers.append(drop)
            prev = drop

        # --- Dense Hidden ---
        dense_hidden = Dense(
            name='Dense_hidden',
            units=dense_units,
            activation='relu',
            prev_layer_or_block=prev,
            wt_init=wt_init
        )
        self.layers.append(dense_hidden)
        prev = dense_hidden

        # --- Output Layer ---
        output = Dense(
            name='Dense_output',
            units=C,
            activation='softmax',
            prev_layer_or_block=prev,
            wt_init=wt_init
        )
        self.layers.append(output)

        self.output_layer = output



class ConvPCN7XL(ConvPCN):
    '''Larger ConvPCN with 7 weight layers/blocks:

    - 1 solo `Conv2D` layer at the start of the network
    - 4 ConvPCNBlocks
    - 1 dense hidden layer
    - 1 output layer

    Here is the specific connectivity:
    Input layer → Conv2D layer → [ConvPCNBlock block]x4 → Flatten → Dropout → Dense hidden → Dense output
    '''
    def __init__(self, input_feats_shape, C, conv_units=64, pcn_units=(128, 128, 256, 256), num_steps=5,
                 step_alpha=0.5, maxpool_in_pcn_block=(False, False, True, True), dense_units=128,
                 dropout_rate=0.2, wt_init='he', do_group_norm=True):
        '''ConvPCN7XL constructor. Builds and configures the PCN, from input to output layers.

        Parameters:
        -----------
        input_feats_shape: tuple of ints.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        C: int.
            Number of classes in the dataset.
        conv_units: int.
            Number of filters the first 2D convolutional layer has in it.
        pcn_units: tuple of int.
            The number of units/filters in each PCN block. The length of the tuple specifies the number of PCNBlocks
            that are used. For example, (64, 128, 192) means there are 3 separate PCNBlocks with the respective number
            of filters therein.
        dense_units: int.
            Number of hidden units in the dense hidden layer of the net (H).
        dropout_rate: float or None.
            The dropout rate used in each `Dropout` layer in the net.
            If `None` passed in, then we do not use dropout in the net.
        maxpool_after_pcn_block: tuple of bool.
            Whether to follow each PCNBlock with a max pooling layer to downscale the spatial resolution. Length must
            match the length of `pcn_units`. If there is a max pooling layer, the stride and window size should both
            be set to 2.
        num_steps: int.
            Number of steps to use to update the block's state during each forward pass thru the block.
        state_lr: float.
            The learning rate used to iteratively update the block's state during each forward pass thru the block.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        do_group_norm. bool:
            Whether to do group normalization in each layer and block of the net.

        Reminders:
        - Add your layers to the layers list.
        - Make sure your output layer is assigned to self.output_layer.
        '''
        super().__init__(input_feats_shape)

        self.layers = []

        prev = None

        # Initial Conv
        conv = Conv2D(
            name='Conv2D_init',
            units=conv_units,
            kernel_size=3,
            strides=1,
            activation='relu',
            prev_layer_or_block=prev,
            wt_init=wt_init,
            do_group_norm=do_group_norm
        )
        self.layers.append(conv)
        prev = conv

        # PCN Blocks
        for i, units in enumerate(pcn_units):
            block = ConvPCNBlock(
                blockname=f'PCNBlock_{i}',
                units=units,
                kernel_size=3,
                strides=1,
                num_steps=num_steps,
                state_lr=step_alpha,
                dropout_rate=None,
                wt_init=wt_init,
                do_group_norm=do_group_norm,
                prev_layer_or_block=prev
            )
            self.layers.append(block)
            prev = block

            if maxpool_in_pcn_block[i]:
                pool = MaxPool2D(
                    name=f'MaxPool_{i}',
                    pool_size=2,
                    strides=2,
                    prev_layer_or_block=prev
                )
                self.layers.append(pool)
                prev = pool

        # Flatten
        flatten = Flatten(
            name='Flatten',
            prev_layer_or_block=prev
        )
        self.layers.append(flatten)
        prev = flatten

        # Dropout
        if dropout_rate is not None:
            drop = Dropout(
                name='Dropout',
                rate=dropout_rate,
                prev_layer_or_block=prev
            )
            self.layers.append(drop)
            prev = drop

        # Dense hidden
        dense_hidden = Dense(
            name='Dense_hidden',
            units=dense_units,
            activation='relu',
            prev_layer_or_block=prev,
            wt_init=wt_init
        )
        self.layers.append(dense_hidden)
        prev = dense_hidden

        # Output
        output = Dense(
            name='Dense_output',
            units=C,
            activation='softmax',
            prev_layer_or_block=prev,
            wt_init=wt_init
        )
        self.layers.append(output)

        self.output_layer = output
