'''conv_layers.py
Convolutional layers in a predictive coding network (PCN)
Yilin Pan, Daniel Lyu
CS 443: Bio-Inspired Learning
'''
import tensorflow as tf

import layers


class Conv2D(layers.Layer):
    '''A 2D convolutional layer'''
    def __init__(self, name, units, kernel_size=(1, 1), strides=1, activation='relu', wt_scale=1e-3,
                 prev_layer_or_block=None, wt_init='normal', do_group_norm=False):
        '''Conv2D layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Conv2D_0). Used for debugging and printing summary of net.
        units: ints.
            Number of convolutional filters/units (K).
        kernel_size: int or tuple. len(kernel_size)=2.
            The horizontal and vertical extent (pixels) of the convolutional filters.
            These will always be the same. For example: (2, 2), (3, 3), etc.
            If user passes in an int, we convert to a tuple. Example: 3 → (3, 3)
        strides: int.
            The horizontal AND vertical stride of the convolution operation. These will always be the same.
            By convention, we use a single int to specify both of them.
        activation: str.
            Name of the activation function to apply in the layer.
        wt_scale: float.
            The standard deviation of the layer weights/bias when initialized according to a standard normal
            distribution ('normal' method).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
            NOTE: Ignore until instructed otherwise.
        do_group_norm. bool:
            Whether to do group normalization in this layer.
            NOTE: Ignore until instructed otherwise.

        TODO: Set the parameters as instance variables. Call the superclass constructor to handle setting instance vars
        the child has in common with the parent class.
        '''

        # Keep me. You can specify kernel size as an int, but this snippet converts to tuple be explicitly account for
        # kernel width and height.
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        super().__init__(layer_name=name, activation=activation, prev_layer_or_block=prev_layer_or_block,
                         do_group_norm=do_group_norm)
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.wt_scale = wt_scale
        self.wt_init = wt_init
        
        self.wts = None
        self.bias = None

    def has_wts(self):
        '''Returns whether the Conv2D layer has weights. This is always true so always return... :)'''
        return True

    def init_params(self, input_shape):
        '''Initializes the Conv2D layer's weights and biases.

        Parameters:
        -----------
        input_shape: Python list. len(input_shape)=4.
            The anticipated shape of mini-batches of input that the layer will process: (B, Iy, Ix, K1).
            K1 is the number of units/filters in the previous layer.

        NOTE: Remember to set your wts/biases as tf.Variables so that we can update the values in the network graph
        during training.
        '''
        N, I_y, I_x, n_chans = input_shape

        if self.wt_init == 'he':
            fan_in = self.kernel_size[0] * self.kernel_size[1] * n_chans
            stddev = self.get_kaiming_gain() * tf.math.sqrt(1.0 / tf.cast(fan_in, tf.float32))
        else:
            stddev = self.wt_scale

        self.wts = tf.Variable(tf.random.normal(shape=(self.kernel_size + (n_chans, self.units)),
                                                stddev=stddev), name=self.layer_name+'_wts')
        self.bias = tf.Variable(tf.zeros(shape=(self.units,)), name=self.layer_name+'_bias')

    def compute_net_input(self, x):
        '''Computes the net input for the current Conv2D layer. Uses SAME boundary conditions.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, Iy, Ix, K1).
            Input from the layer beneath in the network. K1 is the number of units in the previous layer.

        Returns:
        --------
        tf.float32 tensor. shape=(B, Iy, Ix, K2).
            The net_in. K2 is the number of units in the current layer.

        TODO:
        1. This layer uses lazy initialization. This means that if the wts are currently None when we enter this method,
        we should call `init_params` to initialize the parameters!
        2. Compute the convolution using TensorFlow's conv2d function. You can leave the dilations and data_format
        keyword arguments to their default values / you do not need to specify these parameters.

        Helpful link: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d

        NOTE: Don't forget the bias!
        '''
        if self.wts is None:
            self.init_params(x.shape)

        strides = [1, self.strides, self.strides, 1]
        net_in = tf.nn.conv2d(x, self.wts, strides=strides, padding='SAME')
        return net_in + self.bias

    def compute_group_norm(self, net_in, eps=0.001):
        '''Computes group normalization for the input tensor. Group normalization normalizes the activations among
        groups of neurons in a layer for each data point independently.

        Uses `self.num_groups` groups of neurons to perform the normalization. If the user has not set a value for
        `self.num_groups` for the current layer, we set it to the number of units in the layer divided by 8 (rounded).
        In this case when the user did not specify `self.num_groups`, we do not allow the number of groups to drop below
        8.

        (Ignore until later in the semester)

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, Iy, Ix, K).
            Input tensor to be normalized.
        eps: float.
            A small constant added to the standard deviation to prevent division by zero. Default is 0.001.

        Returns:
        --------
        tf.float32 tensor. shape=(B, Iy, Ix, K).
            The normalized tensor with the same shape as the input tensor.
        '''
        B, Iy, Ix, K = net_in.shape

        num_groups = self.num_groups if self.num_groups is not None else min(8, K // 8)

        # Reshape into groups: (B, Iy, Ix, G, K//G)
        x_grouped = tf.reshape(net_in, (B, Iy, Ix, num_groups, K // num_groups))
        # Normalize over spatial dims (1, 2) AND channels within group (4)
        mean = tf.reduce_mean(x_grouped, axis=[1, 2, 4], keepdims=True)
        std = tf.math.reduce_std(x_grouped, axis=[1, 2, 4], keepdims=True)
        x_norm = (x_grouped - mean) / (std + eps)

        # Reshape back and apply learned scale/shift
        x_norm = tf.reshape(x_norm, (B, Iy, Ix, K))
        return self.gn_gain * x_norm + self.gn_bias

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Conv2D layer output({self.layer_name}) shape: {self.output_shape}'


class MaxPool2D(layers.Layer):
    '''A 2D maxpooling layer.'''
    def __init__(self, name, pool_size=(2, 2), strides=1, prev_layer_or_block=None):
        '''MaxPool2D layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        pool_size. tuple. len(pool_size)=2.
            The horizontal and vertical size of the pooling window.
            These will always be the same. For example: (2, 2), (3, 3), etc.
        strides. int.
            The horizontal AND vertical stride of the max pooling operation. These will always be the same.
            By convention, we use a single int to specify both of them.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.

        TODO: Set the parameters as instance variables. Call the superclass constructor to handle setting instance vars
        the child has in common with the parent class.
        '''

        # Keep me. You can specify pool window size as an int, but this snippet converts to tuple be explicitly account
        # for window width and height.
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)

        super().__init__(layer_name=name, activation=None, prev_layer_or_block=prev_layer_or_block,
                         do_group_norm=False)
        self.pool_size = pool_size
        self.strides = strides

    def compute_net_input(self, x):
        '''Computes the net input for the current MaxPool2D layer.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K1).
            Input from the layer beneath in the network. Should be 4D (e.g. from a Conv2D or MaxPool2D layer).
            K1 refers to the number of units/filters in the PREVIOUS layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, K2).
            The net_in. K2 refers to the number of units/filters in the CURRENT layer.

        TODO: Compute the max pooling using TensorFlow's max_pool2d function. You can leave the data_format
        keyword arguments to its default value.

        Helpful link: https://www.tensorflow.org/api_docs/python/tf/nn/max_pool2d
        '''
        return tf.nn.max_pool2d(x, ksize=self.pool_size, strides=self.strides, padding='VALID')
    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'MaxPool2D layer output({self.layer_name}) shape: {self.output_shape}'


class Conv2DTranspose(Conv2D):
    '''2D transposed convolution layer'''
    def __init__(self, name, kernel_size=(1, 1), strides=1, activation='linear', wt_scale=1e-3,
                 prev_layer_or_block=None, wt_init='normal', do_group_norm=False):
        '''Conv2DTranspose layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Conv2DTranspose_0). Used for debugging and printing.
        kernel_size: int or tuple. len(kernel_size)=2.
            The horizontal and vertical extent (pixels) of the convolutional filters.
            These will always be the same. For example: (2, 2), (3, 3), etc.
            If user passes in an int, we convert to a tuple. Example: 3 → (3, 3)
        strides. int.
            The horizontal AND vertical stride of the convolution operation. These will always be the same.
            By convention, we use a single int to specify both of them.
        activation: str.
            Name of the activation function to apply in the layer.
        wt_scale: float.
            The standard deviation of the layer weights/bias when initialized according to a standard normal
            distribution ('normal' method).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
            NOTE: Ignore until instructed otherwise.
        do_group_norm. bool:
            Whether to do group normalization in this layer.
            NOTE: Ignore until instructed otherwise.

        TODO: Call the superclass constructor to handle setting instance vars the child has in common with the parent
        class.

        NOTE: The units will be set during lazy initialization so set units to None for the time being.
        '''
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        super().__init__(name=name, units=None, kernel_size=kernel_size, strides=strides, activation=activation,
                         wt_scale=wt_scale, prev_layer_or_block=prev_layer_or_block, wt_init=wt_init,
                         do_group_norm=do_group_norm)

    def init_params(self, input_shape):
        '''Initializes the Conv2D layer's weights and biases.

        Parameters:
        -----------
        input_shape: Python list. len(input_shape)=4.
            The anticipated shape of mini-batches of input that the layer will process: (B, Iy, Ix, K1).
            K1 is the number of units/filters in the previous layer.

        NOTE: Remember to set your wts/biases as tf.Variables so that we can update the values in the network graph
        during training.
        '''

        N, I_y, I_x, n_chans = input_shape

        if self.wt_init == 'he':
            fan_in = self.kernel_size[0] * self.kernel_size[1] * n_chans
            stddev = self.get_kaiming_gain() * tf.math.sqrt(1.0 / tf.cast(fan_in, tf.float32))
        else:
            stddev = self.wt_scale

        self.wts = tf.Variable(tf.random.normal(shape=(self.kernel_size + (self.units, n_chans)),
                                                stddev=stddev), name=self.layer_name+'_wts')
        self.bias = tf.Variable(tf.zeros(shape=(self.units,)), name=self.layer_name+'_bias')

    def compute_net_input(self, x, units_prev):
        '''Computes the net input for the current Conv2D layer. Uses SAME boundary conditions.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, Iy, Ix, K).
            Input from the *current* layer. K is the number of units in the *current* layer.
        units_prev: int.
            Number of units in the *previous* layer beneath the current one.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, units_prev).
            The net_in. units_prev is the number of units in the *previous* layer beneath the current one.

        TODO:
        0. Because we do not know the units in the layer below when creating the Conv2DTranspose layer, self.units
        should be None during the first forward pass. If this is the case, make sure to set the units to `units_prev`
        (remember this layer is for sending feedback).
        1. This layer uses lazy initialization. This means that if the wts are currently None when we enter this method,
        we should call `init_params` to initialize the parameters!
        2. Compute the convolution using TensorFlow's conv2d_transpose function.
        You can leave the dilations and data_format keyword arguments to their default values / you do not need to
        specify these parameters.
        3. When computing the netIn output shape, remember to account for the stride. When stride >1, this has an
        UPSCALING effect on the image. For example for a 32x32 image that is processed with conv2d_transpose and
        stride 2, the final image size will be 64x64.

        Helpful link: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose

        NOTE: Don't forget the bias!
        '''
        if self.units is None:
            self.units = units_prev
        if self.wts is None:
            self.init_params(x.shape)

        B = tf.shape(x)[0]
        Iy = tf.shape(x)[1]
        Ix = tf.shape(x)[2]

        out_shape = tf.stack([B, Iy * self.strides, Ix * self.strides, self.units])
        strides = [1, self.strides, self.strides, 1]
        net_in = tf.nn.conv2d_transpose(x, self.wts, output_shape=out_shape, strides=strides, padding='SAME')
        return net_in + self.bias

    def __call__(self, x, units_prev):
        '''Do a forward pass thru the layer with mini-batch `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K)
            The input mini-batch computed in the current layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, K_prev)
            The activation computed on the current mini-batch.

        TODO: This should be the same as the one in Layer, except note the difference in method signature for
        compute_net_input.
        '''
        net_in = self.compute_net_input(x, units_prev)
        if self.do_group_norm and self.gn_gain is not None:
            net_in = self.compute_group_norm(net_in)
        net_act = self.compute_net_activation(net_in)
        if self.output_shape is None:
            self.output_shape = list(net_act.shape)
        return net_act

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Conv2DTranspose layer output({self.layer_name}) shape: {self.output_shape}'
