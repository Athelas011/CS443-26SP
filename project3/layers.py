'''layers.py
Neural network layers (e.g. Dense, Dropout, etc.) implemented with the low-level TensorFlow API.
Ariel Pan, Daniel Lyu
CS 443: Bio-Inspired Learning
'''
import tensorflow as tf

class Layer:
    '''Parent class for all specific neural network layers (e.g. Dense, Dropout). Implements all functionality shared in
    common across different layers (e.g. net_in, net_act).
    '''
    def __init__(self, layer_name, activation, prev_layer_or_block, do_group_norm=False):
        '''Neural network layer constructor. You should not generally make Layers objects, rather you should instantiate
        objects of the subclasses (e.g. Dense, Conv2D).

        Parameters:
        -----------
        layer_name: str.
            Human-readable name for a layer (Dense_0, Dense_1, etc.). Used for debugging and printing summary of net.
        activation: str.
            Name of activation function to apply within the layer (e.g. 'relu', 'linear').
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.
        do_group_norm. bool:
            Whether to do group normalization in the layer.
            NOTE: Ignore until instructed otherwise later in the semester.

        TODO: Make instance variables for each of the constructor parameters
        '''
        self.wts = None
        self.bias = None
        self.units = None
        self.output_shape = None

        # We need to make this tf.Variable so this boolean gets added to the static graph when net compiled. Otherwise,
        # bool cannot be updated during training when using @tf.function
        self.is_training = tf.Variable(False, trainable=False)

        # The following relates to features you will implement later in the semester. Ignore for now.
        self.num_groups = None
        self.gn_gain = None
        self.gn_bias = None

        self.layer_name = layer_name
        self.activation = activation
        self.prev_layer_or_block = prev_layer_or_block
        self.do_group_norm = do_group_norm

    def get_name(self):
        '''Returns the human-readable string name of the current layer.'''
        return self.layer_name

    def get_act_fun_name(self):
        '''Returns the activation function string name used in the current layer.'''
        return self.activation

    def get_prev_layer_or_block(self):
        '''Returns a reference to the Layer object that represents the layer below the current one.'''
        return self.prev_layer_or_block

    def get_wts(self):
        '''Returns the weights of the current layer'''
        return self.wts

    def get_b(self):
        '''Returns the bias of the current layer'''
        return self.bias

    def has_wts(self):
        '''Does the current layer store weights? By default, we assume it does not (i.e. always return False).'''
        return self.wts is not None

    def get_num_units(self):
        '''Returns the number of units (neurons) in the layer.'''
        return self.units

    def set_activation_function(self, act_fun_str):
        '''Sets the activation function to the string `act_fun_str`.'''
        self.activation = act_fun_str

    def set_tanh_beta(self, beta):
        '''Sets the β hyperparameter in the tanh activation function to the value `beta`.

        (Week 3)
        NOTE: Ignore until instructed otherwise.
        '''
        self.tanh_beta = beta

    def set_num_groups(self, groups):
        '''Sets the number of normalization groups to use within the layer for group normalization.'''
        self.num_groups = groups

    def get_mode(self):
        '''Returns whether the Layer is in a training state.

        HINT: Check out the instance variables above...
        '''
        return bool(self.is_training.numpy())

    def set_mode(self, is_training):
        '''Informs the layer whether the neural network is currently training. Used in Dropout and some other layer
        types.

        Parameters:
        -----------
        is_training: bool.
            True if the network is currently training, False otherwise.

        TODO: Update the appropriate instance variable according to the state passed into this method.
        NOTE: Notice the instance variable is of type tf.Variable. We do NOT want to use = assignment, otherwise
        TensorFlow will create a new node everything this method is called and the variable in the compiled network
        graph will NOT be updated. Practically, this means:
        1. You will pull your hair out wondering why the training state of the network is NOT being updated, even though
        you pass True into this method :(
        2. You will create a memory leak because TF will many duplicate nodes in the compiled graph (that do nothing).

        Use the `assign` method on the instance variable to update the training state.
        This method should be a one-liner.
        '''
        self.is_training.assign(is_training)

    def init_params(self, input_shape):
        '''Initializes the Layer's parameters (wts + bias), if it has any.

        Leave this parent method empty — subclasses should implement this.
        '''
        pass

    def compute_net_input(self, x):
        '''Computes the net_in on the input tensor `x`.

        Leave this parent method empty — subclasses should implement this.
        '''
        pass

    def compute_net_activation(self, net_in):
        '''Computes the appropriate activation based on the `net_in` values passed in.

        In Project 1, the following activation functions should be supported: 'relu', 'linear', 'softmax'.

        Parameters:
        -----------
        net_in: tf.float32 tensor. shape=(B, ...)
            The net input computed in the current layer.

        Returns:
        --------
        tf.float32 tensor. shape=(B, ...).
            The activation computed on the current mini-batch.

        NOTE:
        - `B` is the batch size.
        - The ... above in the net_in shape refers to the fact that the number of non-batch dimensions could be
        different, depending on the layer (e.g. Dense vs Conv2D). Do NOT write code that makes assumptions about which
        or how many non-batch dimensions are available.
        - To prevent silent bugs, I suggest throwing an error if the user sets an unsupported activation function.
        - Unless instructed otherwise, you may use the activation function implementations provided by the low level
        TensorFlow API here (You already implemented them in CS343 so you have earned it :)
        '''

        if self.activation is None:
            return net_in
        if self.activation == 'relu':
            return tf.nn.relu(net_in)
        elif self.activation == 'linear':
            return net_in
        elif self.activation == 'softmax':
            return tf.nn.softmax(net_in, axis=-1)
        elif self.activation == 'tanh':
            return tf.tanh(net_in * self.tanh_beta)
        else:
            raise ValueError(f'Unsupported activation function {self.activation}. Supported options: "relu", "linear", "softmax".')
        

    def __call__(self, x):
        '''Do a forward pass thru the layer with mini-batch `x`.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, ...)
            The input mini-batch computed in the current layer.

        Returns:
        --------
        tf.float32 tensor. shape=(B, ...).
            The activation computed on the current mini-batch.

        NOTE:
        - `B` is the batch size.
        - The ... above in the net_in shape refers to the fact that the number of non-batch dimensions could be
        different, depending on the layer (e.g. Dense vs Conv2D). Do NOT write code that makes assumptions about which
        or how many non-batch dimensions are available.

        TODO:
        1. Do the forward pass thru the layer (i.e. compute net_in and net_act).
        2. Before the method ends, check to see if `self.output_shape` is None. If it is, that means we are processing
        our very first mini-batch of data ever (e.g. at the beginning of training). If `self.output_shape` is None,
        set it to the shape of the layer's activation, represented as a Python list. You can convert something into a
        Python list by calling the `list` function — e.g. `list(blah)`.
        '''
        net_in = self.compute_net_input(x)
        if self.do_group_norm and self.gn_gain is not None:
            net_in = self.compute_group_norm(net_in)
        net_act = self.compute_net_activation(net_in)
        if self.output_shape is None:
            self.output_shape = list(net_act.shape)
        return net_act

    def get_params(self):
        '''Gets a list of all the parameters learned by the layer (wts, bias, etc.).

        This method is provided to you, so you should not need to modify it.
        '''
        params = []

        if self.wts is not None:
            params.append(self.wts)
        if self.bias is not None and self.bias.trainable:
            params.append(self.bias)
        # The following relates to features you will implement later in the semester. Running code should not
        # affect anything you are implementing now.
        if self.gn_gain is not None:
            params.append(self.gn_gain)
        if self.gn_bias is not None:
            params.append(self.gn_bias)

        return params

    def get_kaiming_gain(self):
        '''Returns the Kaiming gain that is appropriate for the current layer's activation function.

        Returns:
        --------
        float.
            The Kaiming gain.
        '''
        if self.activation == 'relu':
            return tf.math.sqrt(2.0)
        elif self.activation == 'tanh':
            return 5.0 / 3.0
        else:
            return 1.0

    def is_doing_groupnorm(self):
        '''Returns whether the current layer is using group normalization.

        Returns:
        --------
        bool.
            True if the layer has batch normalization turned on, False otherwise.
        '''
        return self.do_group_norm

    def compute_group_norm(self, net_in, eps=0.001):
        '''Computes the group normalization based on on the net input `net_in`.

        Leave this parent method empty — subclasses should implement this.
        '''
        pass

    def init_groupnorm_params(self):
        '''Initializes the parameters for group normalization if group normalization is enabled.

        (Ignore until later in the semester)

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, ...).
            Input tensor to be normalized.

        TODO:
        1. Initialize the group normalization gain and bias instance vars.
        2. Turn off the ordinary bias by replacing the bias with a non-trainable tf.Variable scalar of 0.
        '''
        if not self.do_group_norm:
            return

        self.gn_gain = tf.Variable(tf.ones(shape=(self.units,)), trainable=True)
        self.gn_bias = tf.Variable(tf.zeros(shape=(self.units,)), trainable=True)

        # Turn off ordinary bias
        if self.bias is not None:
            self.bias = tf.Variable(tf.zeros_like(self.bias), trainable=False)


class Dense(Layer):
    '''Neural network layer that uses Dense net input.'''
    def __init__(self, name, units, activation='relu', wt_scale=1e-3, prev_layer_or_block=None, wt_init='normal',
                 do_group_norm=False):
        '''Dense layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Dense_0). Used for debugging and printing summary of net.
        units: int.
            Number of units in the layer (H).
        activation: str.
            Name of activation function to apply within the layer (e.g. 'relu', 'linear').
        wt_scale: float.
            The standard deviation of the layer weights when initialized according to a standard normal distribution
            ('normal' method).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
            NOTE: Ignore until later in the semester.
        do_group_norm. bool:
            Whether to do group normalization in the layer.
            NOTE: Ignore until later in the semester.

        TODO: Set the parameters as instance variables. Call the superclass constructor to handle setting instance vars
        the child has in common with the parent class.
        '''
        super().__init__(layer_name = name, activation=activation, prev_layer_or_block=prev_layer_or_block, do_group_norm=do_group_norm)
        self.units = units
        self.wt_scale = wt_scale
        self.wt_init = wt_init

    def has_wts(self):
        '''Returns whether the Dense layer has weights. This is always true so always return... :)'''
        return True

    def init_params(self, input_shape):
        '''Initializes the Dense layer's weights and biases.

        Parameters:
        -----------
        input_shape: Python list.
            The anticipated shape of mini-batches of input that the layer will process. Starting out, this list will
            look: (B, M).

        NOTE:
        - Remember to set your wts/biases as tf.Variables so that we can update the values in the network graph during
        training.
        - DO NOT assume the number of units in the previous layer is input_shape[1]. Instead, determine it as the last
        element of input_shape. This may sound silly, but doing this will prevent you from having to modify this method
        later in the semester :)
        '''
        input_feats = input_shape[-1]

        if self.wt_init == 'he':
            stddev = self.get_kaiming_gain() * tf.math.sqrt(1.0 / tf.cast(input_feats, tf.float32))
        else:
            stddev = self.wt_scale

        self.wts = tf.Variable(tf.random.normal(shape=(input_feats, self.units), stddev=stddev), trainable=True)
        self.bias = tf.Variable(tf.zeros(shape=(self.units,)), trainable=True)

    def compute_net_input(self, x):
        '''Computes the net input for the current Dense layer.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, M).
            Input from the layer beneath in the network.

        Returns:
        --------
        tf.float32 tensor. shape=(B, H).
            The net_in.

        NOTE: This layer uses lazy initialization. This means that if the wts are currently None when we enter this
        method, we should call `init_params` to initialize the parameters!
        '''
        if self.wts is None:
            self.init_params(x.shape)
        return tf.matmul(x, self.wts) + self.bias

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
        x: tf.float32 tensor. shape=(B, M).
            Input tensor to be normalized.
        eps: float.
            A small constant added to the standard deviation to prevent division by zero. Default is 0.001.

        Returns:
        --------
        tf.float32 tensor. shape=(B, M).
            The normalized tensor with the same shape as the input tensor.
        '''
        B, H = net_in.shape

        num_groups = self.num_groups if self.num_groups is not None else min(8, H // 8)

        # Reshape into groups: (B, G, H//G)
        x_grouped = tf.reshape(net_in, (B, num_groups, H // num_groups))
        mean = tf.reduce_mean(x_grouped, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x_grouped, axis=-1, keepdims=True)
        x_norm = (x_grouped - mean) / (std + eps)

        # Reshape back and apply learned scale/shift
        x_norm = tf.reshape(x_norm, (B, H))
        return self.gn_gain * x_norm + self.gn_bias

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Dense layer output({self.layer_name}) shape: {self.output_shape}'


class Dropout(Layer):
    '''A dropout layer that nixes/zeros out a proportion of the net input signals.'''
    def __init__(self, name, rate, prev_layer_or_block=None):
        '''Dropout layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Dropout_0). Used for debugging and printing summary of net.
        rate: float.
            Proportion (between 0.0 and 1.0.) of net_in signals to drop/nix within each mini-batch.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.

        TODO: Set the parameters as instance variables. Call the superclass constructor to handle setting instance vars
        the child has in common with the parent class.
        '''
        super().__init__(layer_name = name, activation = None, prev_layer_or_block=prev_layer_or_block, do_group_norm=False)
        self.rate = rate


    def compute_net_input(self, x):
        '''Computes the net input for the current Dropout layer.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, ...).
            Input from the layer beneath in the network. This could be 2D (e.g. (B, H)) if the preceding layer is Dense
            or another number of dimensions (e.g. 4D (B, Iy, Ix, K) for Conv2D).

        Returns:
        --------
        tf.float32 tensor. shape=(B, ...), same shape as the input `x`.
            The net_in.

        NOTE:
        - Remember that computing the Dropout net_in operates differently in train and non-train modes.
        - Because the shape of x could be variable in terms of the number of dimensions (e.g. 2D, 4D), do not hard-code
        axes when working with shapes. For example, blah.shape[2] is considered hard coding because blah may not always
        have an axis 2.
        '''
        if self.get_mode():
            # Trainable mode
            keep_prob = 1.0 - self.rate
            return tf.nn.dropout(x, rate=self.rate)
        else:
            return x

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Dropout layer output({self.layer_name}) shape: {self.output_shape}'


class Flatten(Layer):
    '''A flatten layer that flattens the non-batch dimensions of the input signal.'''
    def __init__(self, name, prev_layer_or_block=None):
        '''Flatten layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.

        TODO: Set the parameters as instance variables. Call the superclass constructor to handle setting instance vars
        the child has in common with the parent class.
        '''
        super().__init__(layer_name = name, activation=None, prev_layer_or_block=prev_layer_or_block, do_group_norm=False)
        self.output_shape = None

    def compute_net_input(self, x):
        '''Computes the net input for the current Flatten layer.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, ...).
            Input from the layer beneath in the network. Usually the input will come from Conv2D or MaxPool2D layers
            in which case the shape of `x` is 4D: (B, Iy, Ix, K).

        Returns:
        --------
        tf.float32 tensor. shape=(B, F),
            The net_in. Here `F` is the number of units once the non-batch dimensions of the input signal `x` are
            flattened out.

        NOTE:
        - While the shape of the input `x` will usually be 4D, it is better to not hard-code this just in case.
        For example, do NOT do compute the number of non-batch inputs as x.shape[1]*x.shape[2]*x.shape[3]
        '''
        B = int(x.shape[0])
        self.output_shape = [B, int(tf.reduce_prod(x.shape[1:]).numpy())]
        return tf.reshape(x, (B, -1))

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Flatten layer output({self.layer_name}) shape: {self.output_shape}'
