'''skipgram_layers.py
New neural network layers used for the Skipgram Network
Daniel Lyu, Ariel Pan
CS 443: Bio-Inspired Machine Learning
Project 3: Word Embeddings and Self-Organizing Maps (SOMs)
'''
import tensorflow as tf

from layers import Dense

class Embedding(Dense):
    '''An Embedding layer, which is just like a regular Dense layer, except it interprets each mini-batch as
    a collection of INDICES to select or pull out rows of the Dense weight matrix. These extracted rows extracted from
    the weight matrix are net_in/net_act (after adding the bias).

    An Embedding layer uses linear activation and He initialization.
    '''
    def __init__(self, name, units, prev_layer_or_block=None):
        '''Embedding layer constructor

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Embed_0). Used for debugging and printing summary of net.
        units. int.
            The number of neurons in the layer H.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.

        You should only need to call and pass in relevant information into the superclass constructor to implement this
        method.
        '''
        super().__init__(name=name, units=units, activation='linear', prev_layer_or_block=prev_layer_or_block,wt_init='he')

    def compute_net_input(self, x):
        '''Computes the net input for the current Embedding layer.

        (This method is mostly provided to you, except see the one todo item below)

        Parameters:
        -----------
        x: tf.int32 tensor. shape=(B,).
            Mini-batch of indices of weights to extract for the net input computation.
            NOTE: On 1st compile 'pilot' forward pass, the shape is actually (1, vocab_sz) to make sure the weights get
            initialized correctly.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, H).
            The net_in.

        NOTE: Don't forget the bias!
        '''
        # KEEP THE FOLLOWING
        if self.wts is None:
            self.init_params(input_shape=x.shape)
            # Special case to only handle lazy wt/bias initialization during net compile
            return x @ self.wts + self.bias  # Handles B and T batching automatically. Yay!

        # TODO: Handle regular (non-pilot) case during training/inference where `x` has shape (B,) and contains
        # INDICIES to extract corresponding embedding layer weights.
        # Don't forget about the bias!
        else:
            net_in = tf.gather(self.wts, x) + self.bias
        
        return net_in
