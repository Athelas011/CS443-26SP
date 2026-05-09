'''dense_pcn_layer.py
Densely connected layers in a predictive coding network (PCN)
Daniel Lyu, Ariel Pan
CS 443: Bio-Inspired Learning
'''
import tensorflow as tf

from layers import Dense


class PCNLayer(Dense):
    '''Parent class for densely connected predictive coding network layers.

    Child classes differ based on their position in the network:
    - Input layer (`InputPCNLayer`)
    - Hidden layer (`DensePCNLayer`)
    - Output layer (`OutputPCNLayer`)

    This class contains computations that all layer subtypes have in common.
    '''
    def __init__(self, name, units, activation, wt_scale=1e-2, prev_layer_or_block=None, gamma_lr=0.1,
                 next_layer_or_block=None):
        '''PCNLayer Constructor

        Parameters:
        -----------
        name: str.
            Human-readable name for a layer. Used for debugging and printing summary of net.
        activation: str.
            Name of activation function to apply within the layer. Assumed to be 'linear'.
        wt_scale: float.
            The standard deviation of the layer weights when initialized according to a standard normal distribution
            ('normal' method).
        prev_layer_or_block: PCNLayer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.
        gamma_lr: float.
            The strength with which the predictive feedforward and feedback signals affect the layer's evolving state.
        next_layer_or_block: PCNLayer (or Layer-like) object.
            Reference to the PCNLayer object that is on top the current PCNLayer object. `None` if there is no next
            layer.
            Example InputPCNLayer → DensePCNLayer_1 → DensePCNLayer_2 → OutputPCNLayer.
                For the object DensePCNLayer_1 `next_layer_or_block` is DensePCNLayer_2.

        TODO:
        1. Call the `Dense` layer superclass constructor, passing in relevant parameters that it already handles.
        2. Create additional instance variables as needed.
        '''
        
        super().__init__(name, units, activation, prev_layer_or_block = prev_layer_or_block)
        #if prev_layer_or_block != None:
            #print(prev_layer_or_block.get_name())
        # Keep the following
        self.is_clamped = tf.Variable(False, trainable=False)  # Is the state protected from evolving from feedback?
        self.units = units
        self.wt_scale = wt_scale
        self.gamma_lr = gamma_lr
        self.next_layer_or_block = next_layer_or_block
        # Medium term memory / state of the layer's activations shaped by the prediction errors
        self.state = None  # Will be a tf.float32 tensor of shape (B, H)

    def get_prev_layer(self):
        '''Returns a reference to the PCNLayer object that represents the layer below the current one.'''
        return self.prev_layer_or_block

    def get_next_layer(self):
        '''Returns a reference to the PCNLayer object that represents the layer above the current one.'''
        return self.next_layer_or_block

    def set_next_layer(self, next_layer):
        '''Assign a reference to the specified `PCNLayer` object `next_layer` as the next layer in the PCN.

        Parameters:
        -----------
        next_layer: PCNLayer (or Layer-like) object.
            The next layer in the network.
        '''
        
        self.next_layer_or_block = next_layer
    

    def is_output_layer(self):
        '''Returns whether the current layer is an output layer. By default, this is false here.'''
        if self.get_next_layer() == None and "output" in self.layer_name:
            return True
        else:
            return False

    def get_state(self):
        '''Returns the evolving state maintained by the layer.

        Returns:
        --------
        tf.float32 tensor. Shape=(B, H).
            The layer's evolving state.
        '''
        return self.state

    def reset_state(self, B):
        '''Zero out the layer state.

        Parameters:
        -----------
        B: int.
            The current mini-batch size.
        '''
        self.state = tf.Variable(tf.zeros((B, self.units)), trainable=False)

    def set_state(self, state):
        '''Assigns the new state `state` as the layer's evolving state, replacing anyone that previously exists.'''
        self.state.assign(state)

    def clamp_state(self):
        '''Modify the `is_clamped` instance variable to indicate that the state is now clamped (i.e. cant be modified).
        Use the .assign method (not = operator)
        '''
        self.is_clamped.assign(True)
    def unclamp_state(self):
        '''Modify the `is_clamped` instance variable to indicate that the state is NOT clamped (i.e. CAN be modified).
        Use the .assign method (not = operator).
        '''
        self.is_clamped.assign(False)


    def __call__(self, x):
        '''Do a forward pass thru the layer with mini-batch `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, M)
            The input mini-batch computed in the current layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, H)
            The activation computed on the current mini-batch.

        TODO:
        1. Copy-paste for implementation from `Layer`.
        2. Before the method ends, assign the netAct to the state instance variable.
        '''
         # Linear part
        
        # Compute net input (this also lazily initializes weights)
        net_in = self.compute_net_input(x)
        # Apply activation
        netAct = self.compute_net_activation(net_in)

        # Store activation as the layer state
        self.state = tf.Variable(netAct)

        return netAct

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'{self.layer_name} ({self.units} units). Output shape: {self.output_shape}'


class InputPCNLayer(PCNLayer):
    '''Dense PCN input layer. Represents the current mini-batch as well as the state, which deviates from the mini-batch
    data due the feedback from the PCNLayer above only when the layer is unclamped.
    '''
    def __init__(self, name, units, gamma_lr=0.1, next_layer_or_block=None):
        '''InputPCNLayer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for a layer. Used for debugging and printing summary of net.
        units: int.
            Number of input data features.
        wt_scale: float.
            The standard deviation of the layer weights when initialized according to a standard normal distribution
            ('normal' method).
        gamma_lr: float.
            The strength with which the predictive feedforward and feedback signals affect the layer's evolving state.
        next_layer_or_block: PCNLayer (or Layer-like) object.
            Reference to the PCNLayer object that is on top the current PCNLayer object. `None` if there is no next
            layer.

        TODO:
        1. Call the `PCNLayer` layer parent class constructor, passing in relevant parameters that it already handles.
        2. Clamp the layer state by default.
        '''

        super().__init__(name,
                         units,
                         activation='linear',
                         prev_layer_or_block=None,
                         gamma_lr=gamma_lr,
                         next_layer_or_block=next_layer_or_block)
        self.is_clamped = tf.Variable(False, trainable=False)
        # Clamp the state by default
        self.clamp_state()

        # Keep the following to be used later on
        self.mask = None

    def has_wts(self):
        '''Returns whether the layer has weights. Input layers do NOT have weights, so... :)'''
        return False

    def compute_net_input(self, x):
        '''Computes the "net input" for the input layer. There is no true "net input" computation so this is just the
        mini-batch of data `x`...

        Parameters:
        -----------
        tf.constant. tf.float32s. shape=(B, M)
            The mini-batch of data.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, M)
            The mini-batch of data.
        '''
        return x

    def set_mask(self, mask):
        '''Assigns the occlusion mask `mask` for the current mini-batch to the corresponding instance variable.

        (Ignore this until instructed otherwise)

        Parameters:
        -----------
        tf.constant. tf.float32s. shape=(B, M)
            The occlusion mask for the current mini-batch of data.
        '''
        self.mask = mask

    def update_state(self):
        '''Updates the state in the input layer based on predictive feedback from the PCNLayer above.

        TODO:
        1. Compute the prediction error, which is the difference from the layer's current state and the layer above's
        predicted input.
        2. Update the layer's state. Refer to the notebook for a refresher on the equation.

        NOTE: We ONLY update the state (or do any work) when the input layer is NOT clamped.
        '''

        # Only update when the layer is NOT clamped
        if self.is_clamped:
            return

        # Predicted input from the layer above
        pred = self.get_next_layer().predict_input()

        # Prediction error
        error = self.state - pred

        # If a mask is set, zero out the prediction error where mask=1 (real image data).
        # This confines state updates to only the occluded/masked regions (mask=0).
        if self.mask is not None:
            error = error * (1 - self.mask)

        # Update state
        self.state.assign(self.state - self.gamma_lr * error)


class DensePCNLayer(PCNLayer):
    '''Dense PCN hidden (internal) layer.

    This layer type:
    - has weights.
    - has a PCN layer both below and above the current one.
    - computes prediction errors based on its state and the state of the layer below.
    - updates the state based on the bottom-up input from the layer below and top-down feedback from the layer above.
    '''
    def __init__(self, name, units, wt_scale=1e-2, prev_layer_or_block=None, gamma_lr=0.1, next_layer_or_block=None):
        '''DensePCNLayer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for a layer. Used for debugging and printing summary of net.
        units: int.
            Number of hidden units in the layer
        wt_scale: float.
            The standard deviation of the layer weights when initialized according to a standard normal distribution
            ('normal' method).
        prev_layer_or_block: PCNLayer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        gamma_lr: float.
            The strength with which the predictive feedforward and feedback signals affect the layer's evolving state.
        next_layer_or_block: PCNLayer (or Layer-like) object.
            Reference to the PCNLayer object that is on top the current PCNLayer object. `None` if there is no next
            layer.

        TODO: Call the `PCNLayer` layer parent class constructor, passing in relevant parameters.
        '''
        super().__init__(name,
                         units,
                         activation='linear',
                         wt_scale=wt_scale,
                         prev_layer_or_block=prev_layer_or_block,
                         gamma_lr=gamma_lr,
                         next_layer_or_block=next_layer_or_block)

    def init_params(self, input_shape):
        '''Initializes the layer's weights. This is the same as the method in `Dense`, except we turn off the bias.

        Parameters:
        -----------
        input_shape: Python list. Example: (B, M)
            The anticipated shape of mini-batches of input that the layer will process.

        TODO:
        1. Call the `Dense` parent version of this method to avoid code redundancy.
        2. Turn off the bias by assigning it to None.
        '''
        super().init_params(input_shape)
        self.bias = None

    def compute_net_input(self, x):
        '''Computes the net input for the current dense PCNLayer.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, M).
            Input from the layer beneath in the network.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, H).
            The net_in.

        TODO: Copy-paste from your `Dense` implementation but adapt for the fact that this layer has no bias.
        '''
        if self.wts is None:
            self.init_params(x.shape)

        return tf.matmul(x, self.wts)

    def predict_input(self):
        '''Use the layer evolving state and the weights to predict the expected bottom-up input to the current layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, M) or (B, H_prev) if this is hidden layer 2+.
            The predicted input.
        '''
        # Done iteratively
        # We will compute this hidden layer 1 -> output layers
        input_pred = self.state @ tf.transpose(self.wts)
        return input_pred

    def prediction_error(self):
        '''Computes the prediction error, which is the difference between the layer below's state and the current
        layer's predicted input.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, M) or (B, H_prev) if this is hidden layer 2+.
            The prediction error.
        '''
        prev_layer = self.get_prev_layer()
        return prev_layer.get_state() - self.predict_input()

    def update_state(self):
        '''Updates the layer's state using gradients based on:
        - the current layer's prediction error (bottom-up)
        - the next layer's prediction error (top-down)

        Refer to the notebook for a refresher on the equation.
        '''

        # bottom-up prediction error
        eps = self.prediction_error()

        # top-down prediction error
        next_layer = self.get_next_layer()
        eps_next = next_layer.prediction_error()

        # bottom-up term
        bottom_up = tf.matmul(eps, self.wts)

        # top-down term
        top_down = eps_next

        # update state
        self.state.assign(self.state + self.gamma_lr * (bottom_up - top_down))


class OutputPCNLayer(DensePCNLayer):
    '''Dense PCN output layer.

    This layer is the same as `DensePCNLayer` except its state is only updated because on the "bottom-up gradient"
    (the layer is output layer after all!!).
    '''
    def __init__(self, name, units, wt_scale=1e-2, prev_layer_or_block=None, gamma_lr=0.1):
        '''OutputPCNLayer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for a layer. Used for debugging and printing summary of net.
        units: int.
            Number of units in the output layer.
        wt_scale: float.
            The standard deviation of the layer weights when initialized according to a standard normal distribution
            ('normal' method).
        prev_layer_or_block: PCNLayer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        gamma_lr: float.
            The strength with which the predictive feedforward and feedback signals affect the layer's evolving state.

        TODO: Call the parent class constructor, passing in relevant parameters.
        '''
        super().__init__(name,
                         units,
                         wt_scale=wt_scale,
                         prev_layer_or_block=prev_layer_or_block,
                         gamma_lr=gamma_lr,
                         next_layer_or_block=None)

    def is_output_layer(self):
        '''Returns whether this is an output layer.

        (This is provided and does not require modification :)
        '''
        return True

    def update_state(self):
        '''Updates the layer's state using gradients based on the current layer's prediction error (bottom-up).

        Refer to the notebook for a refresher on the equation.

        NOTE: We make no updates to the state if the layer is CLAMPED.
        '''
        if self.is_clamped:
            return

        # bottom-up prediction error
        eps = self.prediction_error()

        # bottom-up gradient
        bottom_up = eps @ self.wts

        # update state
        self.state.assign(self.state + self.gamma_lr * bottom_up)