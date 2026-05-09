'''block.py
Defines the parent Block class
Oliver W. Layton
CS 443: Bio-Inspired Machine Learning
'''


class Block:
    '''The `Block` parent class and specifies functionality shared by all blocks. All blocks inherit from this class.'''
    def __init__(self, blockname, prev_layer_or_block):
        '''Block constructor.

        This method is provided to you, so you should not need to modify it.

        Parameters:
        -----------
        blockname: str.
            Human-readable name for a block (ConvPCNBlock_0, ConvPCNBlock_1, etc.). Used for debugging and printing
            summary of net.
        prev_layer_or_block: Layer or Block object.
            Reference to the Layer or Block object that is beneath the current Layer object. `None` if there is no
            preceding layer or block.
            Example: ConvPCNBlock_0 → ConvPCNBlock_1 → Flatten → ...
                The ConvPCNBlock_1 block object has `prev_layer_or_block=ConvPCNBlock_0` block.
                The ConvPCNBlock_0 block object has `prev_layer_or_block=None` since it is the first layer or block in
                the net.
        '''
        self.blockname = blockname
        self.prev_layer_or_block = prev_layer_or_block

        self.layers = []

    def get_prev_layer_or_block(self):
        '''Returns a reference to the Layer object that represents the layer/block below the current one.

        This method is provided to you, so you should not need to modify it.
        '''
        return self.prev_layer_or_block

    def get_layer_names(self):
        '''Returns a list of human-readable string names of the layers that belong to this block.

        This method is provided to you, so you should not need to modify it.
        '''
        names = []
        for layer in self.layers:
            names.append(layer.get_name())
        return names

    def get_params(self):
        '''Returns a list of trainable parameters spread out across all layers that belong to this block.

        This method is provided to you, so you should not need to modify it.
        '''
        all_params = []

        for layer in self.layers:
            params = layer.get_params()
            all_params.extend(params)

        return all_params

    def get_wts(self):
        '''Returns a list of trainable weights (no biases/other) spread out across all layers that belong to this block.

        This method is provided to you, so you should not need to modify it.
        '''
        all_wts = []

        for layer in self.layers:
            wts = layer.get_wts()

            if wts is not None:
                all_wts.append(wts)

        return all_wts

    def get_mode(self):
        '''Gets the mode of the block (i.e. training, not training). Since this is always the same in all layers,
        we use the first layer in the block as a proxy for all of them.

        This method is provided to you, so you should not need to modify it.
        '''
        return self.layers[0].get_mode()

    def set_mode(self, is_training):
        '''Sets the mode of every layer in the block to the bool value `is_training`.

        This method is provided to you, so you should not need to modify it.
        '''

        for layer in self.layers:
            layer.set_mode(is_training)

    def init_groupnorm_params(self):
        '''Initializes the group norm parameters in every layer in the block (only should have an effect on them if they
        are configured to perform group normalization).

        This method is provided to you, so you should not need to modify it.
        '''
        for layer in self.layers:
            layer.init_groupnorm_params()

    def __str__(self):
        '''The toString method that gets a str representation of the layers belonging to the current block. These layers
        are indented for clarity.

        This method is provided to you, so you should not need to modify it.
        '''
        string = self.blockname + ':'
        for layer in reversed(self.layers):
            string += '\n\t' + layer.__str__()
        return string
