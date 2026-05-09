'''dense_pcn.py
Densely connected predictive coding network (PCN)
YOUR NAMES HERE
CS 443: Bio-Inspired Learning
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython import display

import network
from dense_pcn_layer import InputPCNLayer, DensePCNLayer, OutputPCNLayer

class DensePCN(network.DeepNetwork):
    '''Predictive coding network with any number of densely connected layers. Uses linear activations throughout.
    '''
    def __init__(self, input_feats_shape, C, hidden_units=(256,), wt_scale=1e-2, gamma_lr=0.1,
                 train_num_steps=20, test_num_steps=10):

        '''DensePCN constructor. Builds the PCN, from input to output layers, and connects successive layers
        bidirectionally to one another.
        '''

        # 1. Call superclass constructor
        super().__init__(input_feats_shape)

        self.train_num_steps = train_num_steps
        self.test_num_steps = test_num_steps

        self.layers = []

        M = input_feats_shape[0]

        # 2. Build input layer
        input_layer = InputPCNLayer("InputLayer", M)
        self.layers.append(input_layer)

        prev = input_layer

        # Hidden layers
        for i, H in enumerate(hidden_units):

            hidden = DensePCNLayer(
                name=f"PredLayer_{i}",
                units=H,
                wt_scale=wt_scale,
                prev_layer_or_block=prev,
                gamma_lr=gamma_lr
            )

            prev.next_layer_or_block = hidden

            self.layers.append(hidden)

            prev = hidden

        # Output layer
        output = OutputPCNLayer(
            name="OutputLayer",
            units=C,
            wt_scale=wt_scale,
            prev_layer_or_block=prev,
            gamma_lr=gamma_lr
        )

        prev.next_layer_or_block = output

        self.layers.append(output)

        self.output_layer = output

    def set_test_num_steps(self, num_steps):
        '''Set method to update the number of steps/iterations/sweeps used by the PCN during inference.
        '''
        self.test_num_steps = num_steps

    def __call__(self, x):
        '''Perform the forward pass through the network.
        '''
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def update_states(self, num_steps, x_batch, y_batch=None):
        '''Performs `num_steps` forward sweeps through the network to update the state in each successive layer.
        '''

        B, M = x_batch.shape

        # 1. Reset the state in all layers
        for layer in self.layers:
            layer.reset_state(B)

        # 2. Perform forward pass to initialize states
        self(x_batch)

        # 3. Configure output clamping
        if y_batch is not None:

            C = self.output_layer.get_num_units()
            yh = tf.one_hot(y_batch, C)

            self.output_layer.state.assign(yh)
            self.output_layer.is_clamped.assign(True)

        else:
            self.output_layer.is_clamped.assign(False)

        # 4. Iteratively update the state
        for _ in range(num_steps):

            for layer in self.layers:
                layer.update_state()

    def loss(self):
        '''Computes the loss for the current minibatch based on the states store in each network layer.
        '''
        if self.loss_name == 'predictive':

            loss = 0

            for layer in self.layers[1:]:
                eps = layer.prediction_error()

                B = tf.shape(eps)[0]  # minibatch size

                loss += 0.5 * tf.reduce_sum(tf.square(eps)) / tf.cast(B, tf.float32)

            return loss

        else:
            raise ValueError(f'Unknown loss function {self.loss_name}')

    def train_step(self, x_batch, y_batch):
        '''Completely process a single mini-batch of data during training.
        '''

        self.set_layer_training_mode(True)

        # update network states
        self.update_states(self.train_num_steps, x_batch, y_batch)

        # compute loss
        loss = self.loss()

        return loss

    def test_step(self, x_batch, y_batch, num_steps):
        '''Completely process a single mini-batch of data during test/validation time.
        '''

        self.set_layer_training_mode(False)

        # update states (no clamping)
        self.update_states(num_steps, x_batch)

        loss = self.loss()

        # predicted classes from output state
        logits = self.output_layer.get_state()

        preds = tf.argmax(logits, axis=1)
        preds = tf.cast(preds, tf.int32)

        acc = self.accuracy(preds, y_batch)

        return acc, loss
    
    def evaluate(self, x, y, batch_sz=64):
        '''Evaluates the accuracy and loss on the data `x` and labels `y`. Breaks the dataset into mini-batches for you
        for efficiency.

        This method is provided to you, so you should not need to modify it.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, M).
            The complete dataset or one of its splits (train/val/test/dev).
        y: tf.constant. tf.ints32. shape=(B,).
            int-coded labels of samples in the complete dataset or one of its splits (train/val/test/dev).
        batch_sz: int.
            The batch size used to process the provided dataset. Larger numbers will generally execute faster, but
            all samples (and activations they create in the net) in the batch need to be maintained in memory at a time,
            which can result in crashes/strange behavior due to running out of memory.
            The default batch size should work fine throughout the semester and its unlikely you will need to change it.

        Returns:
        --------
        float.
            The accuracy.
        float.
            The loss.
        '''
        # Set the mode in all layers to the non-training mode
        self.set_layer_training_mode(is_training=False)

        # Make sure the mini-batch size isn't larger than the number of available samples
        N = len(x)
        if batch_sz > N:
            batch_sz = N

        num_batches = N // batch_sz

        # Make sure the mini-batch size is positive...
        if num_batches < 1:
            num_batches = 1

        # Process the dataset in mini-batches by the network, evaluating and avging the acc and loss across batches.
        loss = acc = 0
        for b in range(num_batches):
            curr_x = x[b*batch_sz:(b+1)*batch_sz]
            curr_y = y[b*batch_sz:(b+1)*batch_sz]

            curr_acc, curr_loss = self.test_step(curr_x, curr_y, self.test_num_steps)
            acc += curr_acc
            loss += curr_loss
        acc /= num_batches
        loss /= num_batches

        return acc, loss

    def dream_input(self, class_names, num_steps=150, image_dims=(28, 28, 1), input_stddev=1e-2, n_plot_rows=2,
                    eps=1e-8):
        '''Leverages the generative capabilities of the PCN to 'dream' up images that are the expected inputs for each
        output layer class neuron.

        NOTE: Much of this method is provided. Fill in parts of the method based on the inline TODO instructions.

        Parameters:
        -----------
        class_names: Python list of str.
            Names of each class in the dataset.
        num_steps: int.
            Number of test steps to use when dreaming.
        image_dims: tuple of ints. format: (Iy, Ix, n_chans).
            Expanded shape of single images in the dataset.
        input_stddev: float.
            Standard deviation to use when generating random noise in the placeholder for the dreamed input images.
        n_plot_rows: int.
            Number of rows of images in the plot that shows the generated dream image for each class / output neuron.
        eps: float.
            Fudge factor to prevent division by 0 when normalizing the generated images for plotting.
        '''
        M = self.input_feats_shape[0]
        C = self.output_layer.get_num_units()
        n_plot_cols = C // n_plot_rows

        fig, axes = plt.subplots(nrows=n_plot_rows, ncols=n_plot_cols, figsize=(12, 5))
        axes = axes.ravel()

        # Create all possible digits as separate samples in curr mini-batch
        generated_inputs = tf.random.normal(shape=(C, M), stddev=input_stddev)

        # We want to generate one dream image per output neuron. To do this, we create the states for the output layer
        # where the dreaming neuron j has state 1 and the rest have 0 for dream image j.
        # For example, this looks like:
        # [1, 0, 0]
        # [0, 1, 0]
        # [0, 0, 1]
        # if there are 3 classes and so 3 dream images to generate.
        one_hot_vecs = tf.eye(C)

        # TODO 1: Before we dream, reset all the states in the net
        for layer in self.layers:
            layer.reset_state(C)

        # TODO 2: Configure the input layer for dreaming: set the initial state to the random noise patterns and unclamp
        # the layer so the state can evolve.
        input_layer = self.layers[0]
        input_layer.state.assign(generated_inputs)
        input_layer.is_clamped.assign(False)

        # TODO 3: Configure the output layer for dreaming: set the state of each the dreaming neuron to 1 and the rest
        # to 0 and clamp the layer so the network dynamics do not change the output layer state.
        self.output_layer.state.assign(one_hot_vecs)
        self.output_layer.is_clamped.assign(True)

        # Prepare plots for fast updating
        img_objs = []
        for i in range(C):
            img_objs.append(axes[i].imshow(np.zeros(image_dims), cmap='gray', vmin=0, vmax=1))
            axes[i].set_title(f'Class {class_names[i]}')
            axes[i].axis('off')

        # We want to do N_steps prediction passes for each class one hot vector
        # This is somewhat like update_states but we DO NOT want to do the forward pass
        # because the input is random noise and we don't care about the net_acts to it.
        # We want this to be 100% top-down driven.
        # Now evolve the states of each layer driven by the output neuron
        for t in range(num_steps):
            # Update the state in each layer for some number of steps to minimize the prediction error
            # We do this in reverse order for speed
            for layer in reversed(self.layers):
                layer.update_state()

            # Animate
            for i in range(C):
                # Reshape the state for the specific digit in the batch
                digit_pixels = tf.reshape(self.layers[0].get_state()[i], image_dims)
                # Min-max normalize for visualization
                min = tf.reduce_min(digit_pixels)
                max = tf.reduce_max(digit_pixels)
                digit_pixels = (digit_pixels - min) / (max - min + eps)
                img_objs[i].set_data(digit_pixels.numpy())

            # Set the suptitle as the frame number
            fig.suptitle(f'Frame {t}')

            display.clear_output(wait=True)
            display.display(fig)
            time.sleep(0.1)

        plt.close()

    def complete_input(self, x_batch, x_mask, y_batch, num_steps=150, image_dims=(28, 28, 1), n_plot_rows=2, eps=1e-8):
        '''Leverages the generative capabilities of the PCN to 'complete' masked portions of images based on the
        top-down prediction signals developed by the network.

        NOTE: Much of this method is provided. Fill in parts of the method based on the inline TODO instructions.

        Parameters:
        -----------
        x_batch: tf.constant. tf.float32s. shape=(B, M).
            A single mini-batch of masked images.
        x_mask: tf.constant. tf.float32s. shape=(B, M).
            The occlusion masks that specify where pixels in each mini-batch image are blanked out.
            Mask format: 0 if corresponding pixel in `x_batch` images is blanked out. 1 if corresponding pixel in
            `x_batch` images contains image information.
        y_batch: tf.constant. tf.ints32. shape=(B,).
            int-coded labels of samples in the mini-batch.
        image_dims: tuple of ints. format: (Iy, Ix, n_chans).
            Expanded shape of single images in the dataset.
        n_plot_rows: int.
            Number of rows of images in the plot that shows each completed image in the mini-batch.
        eps: float.
            Fudge factor to prevent division by 0 when normalizing the completed images for plotting.
        '''
        B = len(x_batch)
        C = self.output_layer.get_num_units()
        n_plot_cols = B // n_plot_rows

        fig, axes = plt.subplots(nrows=n_plot_rows, ncols=n_plot_cols, figsize=(12, 5))
        axes = axes.ravel()

        # Create one hot coding of every class in the mini-batch
        yh_batch = tf.one_hot(y_batch, C)

        # TODO 1: Before we fill in image detail, reset all the states in the net
        for layer in self.layers:
            layer.reset_state(B)


        # TODO 2: Configure the input layer for completion: set the initial state to the masked mini-batch,
        # make the mask available to the layer, and unclamp the layer.
        input_layer = self.layers[0]
        input_layer.state = tf.Variable(x_batch)
        input_layer.mask = x_mask
        input_layer.is_clamped.assign(False)


        # TODO 3: Configure the output layer for completion: set the state to the one-hot class labels and clamp the
        # state
        self.output_layer.state = tf.Variable(yh_batch)
        self.output_layer.is_clamped.assign(True)


        # Prepare plots for fast updating
        img_objs = []
        for i in range(C):
            img_objs.append(axes[i].imshow(np.zeros(image_dims), cmap='gray', vmin=0, vmax=1))
            axes[i].set_title(f'Class {y_batch[i]}')
            axes[i].axis('off')

        # We want to do N_steps prediction passes for each class one hot vector
        # This is somewhat like update_states but we DO NOT want to do the forward pass
        # because the input is random noise and we don't care about the net_acts to it.
        # We want this to be 100% top-down driven.
        # Now evolve the states of each layer driven by the output neuron
        for t in range(num_steps):
            # Update the state in each layer for some number of steps to minimize the prediction error
            # We do this in reverse order for speed
            for layer in reversed(self.layers):
                layer.update_state()

            # Animate
            for i in range(C):
                # Reshape the state for the specific digit in the batch
                digit_pixels = tf.reshape(self.layers[0].get_state()[i], image_dims)
                # Min-max normalize for visualization
                min = tf.reduce_min(digit_pixels)
                max = tf.reduce_max(digit_pixels)
                digit_pixels = (digit_pixels - min) / (max - min + eps)
                img_objs[i].set_data(digit_pixels.numpy())

            # Set the suptitle as the frame number
            fig.suptitle(f'Frame {t}')

            display.clear_output(wait=True)
            display.display(fig)
            time.sleep(0.1)

        plt.close()

