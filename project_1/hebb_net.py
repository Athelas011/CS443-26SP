'''hebb_net.py
Bio-inspired neural network that implements the Hebbian learning rule and competition among neurons in the network
Yilin Pan, Daniel Lyu
CS 443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning
'''
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display

from viz import draw_grid_image


class HebbNet:
    '''Single layer bio-inspired neural network in which neurons compete with each other and learning occurs via a
    competitive variant of a Hebbian learning rule (Oja's Rule).

    NOTE: This network should once again be implemented in 100% TensorFlow, except where noted below.
    '''
    def __init__(self, num_features, num_neurons, k=6, inhib_value=-0.4, load_wts=False, saved_wts_path='export/wts.npy'):
        '''Hebbian network constructor

        Parameters:
        -----------
        num_features: int.
            Num input features (M).
        num_neurons: int.
            Num of neurons in the network (H).
        k: int.
            In the neural competition that occurs when processing each data sample, the neuron that achieves the kth
            highest net_in value ("neuron came in kth place") is inhibited, which means the kth place neuron gets netAct
            value of `-inhib_value`.
        inhib_value: float.
            Non-positive number (≤0) that represents the netAct value assigned to the inhibited neuron (with the kth
            highest netAct value).
        load_wts: bool.
            Whether to load weights previously saved off by the network after successful training.
        saved_wts_path: str.
            Path from the working project directory where the weights previously saved by the net are stored.
            Used if `load_wts` is True.

        TODO:
        - Create instance variables for the parameters
        - Initialize the wts
            - If loading wts, set the wts by loading the previously saved .npy wt file.
            Use `np.load` (this use of NumPy is allowed).
            - Otherwise, initialize the network wts as a tensor containing values sampled from a standard normal
            distribution (stddev = 1.0). shape=(M, H). Should NOT be a `tf.Variable` because we are not tracking
            gradients here.
        '''
        if load_wts:
            self.wts = tf.constant(np.load(saved_wts_path))
            print('Loaded stored wts.')
        else:
            self.wts = tf.Variable(tf.random.normal([num_features, num_neurons], dtype=tf.float32))
  
        self.inhib_value = inhib_value
        self.k = k
        self.saved_wts_path = saved_wts_path

    def get_wts(self):
        '''Returns the Hebbian network wts'''
        return self.wts

    def set_wts(self, wts):
        '''Replaces the Hebbian network weights with `wts` passed in as a parameter.

        Parameters:
        -----------
        wts: tf.float32 tensor. shape=(M, H).
            New Hebbian network weights.
        '''
        self.wts = wts

    def net_in(self, x):
        '''Computes the Hebbian network Dense net_in based on the data `x`.

        Parameters:
        -----------
        x: ndarray. shape=(B, M)

        Returns:
        -----------
        tf.float32 tensor. shape=(B, H).
            netIn
        '''
        if not tf.is_tensor(x):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        else:
            x = tf.cast(x, tf.float32)

        return tf.matmul(x, self.wts)


    def net_act(self, net_in):
        '''Compute the Hebbian network activation, which is a function that reflects competition among the neurons
        based on their net_in values.

        NetAct (also see notebook):
        - 1 for neuron that achieves highest net_in value to sample i
        - -delta for neuron that achieves kth highest net_in value to sample i
        - 0 for all other neurons

        Parameters:
        -----------
        net_in: tf.float32 tensor. shape=(B, H).

        Returns:
        -----------
        tf.float32 tensor. shape=(B, H).
            net_act

        NOTE:
        - It may be helpful to think of this as a two step process: dealing with the winner and then kth place inhibition
        separately.
        - It may be helpful to think of competition as an assignment operation. But you may not be able to use []
        indexing. Instead, refer to your work in the competitive_hebb notebook for a TensorFlow substitute for
        arange indexing.
        - += is a valid TensorFlow operator.
        '''
 
        B = tf.shape(net_in)[0]
        H = tf.shape(net_in)[1]

        # find winner
        winner_idx = tf.argmax(net_in, axis=1, output_type=tf.int32)
        winner_one_hot = tf.one_hot(winner_idx, depth=H, dtype=tf.float32)

        # find kth place neuron
        _, top_indices = tf.math.top_k(net_in, k=self.k)
        kth_idx = top_indices[:, self.k - 1]
        kth_one_hot = tf.one_hot(kth_idx, depth=H, dtype=tf.float32)


        net_act = winner_one_hot + (kth_one_hot * self.inhib_value)
        
        return net_act

    



    def update_wts(self, x, net_in, net_act, lr, eps=1e-10):
        '''Update the Hebbian network wts according to a modified Hebbian learning rule (competitive Oja's rule).
        After computing the weight change based on the current mini-batch, the weight changes (gradients) are normalized
        by the largest gradient (in absolute value). This has the effect of making the largest weight change equal in
        absolute magnitude to the learning rate `lr`. See notebook for equations.

        Parameters:
        -----------
        net_in: tf.float32 tensor. shape=(B, H)
        net_act: tf.float32 tensor. shape=(B, H)
        lr: float. Learning rate hyperparameter
        eps: float. Small non-negative number used in the wt normalization step to prevent possible division by 0.

        NOTE:
        - This is definitely a scenario where you should the shapes of everything to guide you through and decide on the
        appropriate operation (elementwise multiplication vs matrix multiplication).
        - The `keepdims` keyword argument may be convenient here.
        '''
        
        # Hebbian term 
        term1 = tf.matmul(x, net_act, transpose_a=True)   # (M, H)

        # Oja competition term 
        competition = tf.reduce_sum(net_in * net_act, axis=0, keepdims=True)  # (1, H)
        term2 = self.wts * competition  # (M, H)

        # Weight change
        delta_w = term1 - term2

        # Normalize by max |weights| 
        max_abs_w = tf.reduce_max(tf.abs(self.wts))

        # Update weights
        self.wts.assign_add(lr * delta_w / (max_abs_w + eps))


    def fit(self, x, epochs=1, mini_batch_sz=500, lr=1e-2, plot_wts_live=False, fig_sz=(9, 9), n_wts_plotted=(10, 10),
            print_every=1, save_wts=True, ds_feat_shape=(32, 32, 3)):
        '''Trains the Competitive Hebbian network on the training samples `x` using unsupervised Hebbian learning
        (without classes y!).

        Parameters:
        -----------
        x: tf.float32 tensor. dtype=tf.float32. shape=(N, M).
            Data samples.
        epochs: int.
            Number of epochs to train the network.
        mini_batch_sz: int.
            Mini-batch size used when training the Hebbian network.
        lr: float.
            Learning rate used with Hebbian weight update rule
        plot_wts_live: bool.
            Whether to plot the weights and update throughout training every `print_every` epochs.
        fig_sz: tuple.
            Dimensions of the plt.figure that is used to visualize the wts (if doing this).
        n_wts_plotted: tuple.
            Grid arrangement of the weights being visualized/plotting (if doing this).
            Example: (10, 10) means wts of 100 (=10*10) neurons are shown in a 10x10 image grid.
        print_every: int.
            How often, in epochs, to print the min and max weight values across the net and draw the wt grid
            (if doing this)
        save_wts: bool.
            Whether to save the Hebbian network wts (to self.saved_wts_path) after training finishes.
        ds_feat_shape: tuple.
            Shape of the original image dataset (WITHOUT batch dimension). Used for wt visualization (if doing this).

        TODO:
        Very similar workflow to usual:
        - If plotting the wts on the current epoch, update the plot (via `draw_grid_image`) to show the current wts
        `print_every` epochs.
        - In each epoch setup mini-batch. You can sample with replacement or without replacement (shuffle) between
        epochs (your choice).
        - Compute forward pass for each mini-batch then update the weights.
        - Print out which epoch we are on `print_every` epochs
        - When training is done, save the wts if `save_wts` is True. Using `np.save` is totally fine here.
        '''
        # Convert ONCE (fast path). Does not break tests; tests may call net_in separately.
        if not tf.is_tensor(x):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        else:
            x = tf.cast(x, tf.float32)

        N = tf.shape(x)[0]

        if plot_wts_live:
            fig = plt.figure(figsize=fig_sz)

        # Compile the per-batch compute path 
        @tf.function
        def _run_batch(x_batch):
            net_in = self.net_in(x_batch)
            net_act = self.net_act(net_in)
            self.update_wts(x_batch, net_in, net_act, lr)

        for e in range(epochs):
            indices = tf.random.shuffle(tf.range(N))
            x_shuffled = tf.gather(x, indices)

            # Iterate over batches
            for i in tf.range(0, N, mini_batch_sz):
                x_batch = x_shuffled[i:i + mini_batch_sz]
                _run_batch(x_batch)

            # Only print/plot every print_every epochs
            if (e % print_every) == 0:
                if plot_wts_live:
                    draw_grid_image(tf.transpose(self.wts),
                                    n_wts_plotted[0], n_wts_plotted[1],
                                    title=f'Net receptive fields (Epoch {e})',
                                    sample_dims=ds_feat_shape)
                    display.clear_output(wait=True)
                    display.display(fig)
                else:
                    print(f'Starting epoch {e}/{epochs}')

        if save_wts:
            print('Saving weights...', end='')
            np.save(self.saved_wts_path, self.get_wts())
            print('Done!')
