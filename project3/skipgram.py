'''skipgram.py
The Skipgram neural network
Daniel Lyu, Ariel Pan
CS 443: Bio-Inspired Machine Learning
Project 3: Word Embeddings and Self-Organizing Maps (SOMs)
'''
import time
import os
import numpy as np
import tensorflow as tf

import network
from layers import Dense
from skipgram_layers import Embedding
from tf_util import arange_index


class Skipgram(network.DeepNetwork):
    '''Skipgram neural network that learns word embeddings. It consists of the following structure:

    Input → Embedding (linear) → Dense (softmax)

    Both the input and output layer have `vocab_sz` units.

    The output layer uses regular softmax activation.

    The layers use He weight initialization.
    '''
    def __init__(self, input_feats_shape, C, embedding_dim=96):
        '''Skipgram constructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: for text data, input_feats_shape=(vocab_sz,).
        C: int.
            Number of classes in the dataset.
        embedding_dim: int.
            The number of units in the Embedding hidden layer (H).

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common.
        2. Build out and configure the Skipgram network.
        '''
        super().__init__(input_feats_shape)

        embed = Embedding(name='Embed_0', units=embedding_dim, prev_layer_or_block=None)
        out = Dense(name='Output_1', units=C, activation='softmax',
                    prev_layer_or_block=embed, wt_init='he')

        self.layers = [embed, out]
        self.output_layer = out

    def __call__(self, x):
        '''Forward pass through the Skipgram with the data samples `x`.

        Parameters:
        -----------
        x: tf.int32 tensor. shape=(B,).
            Data sample/word INDICES.

        Returns:
        --------
        tf.float32 tensor. shape=(B, C).
            Activations produced by the output layer to the data.
        '''
        # Forward pass through the layers list
        # x is (B,) int indices
        for layer in self.layers:
            x = layer(x)
        return x

    def fit(self, x, y, batch_size=256, epochs=10, print_every=1, linear_lr_decay=True, linear_lr_min_lr=1e-5,
            verbose=True):
        '''Trains Skipgram on pairs of context word indices (samples `x`) and target word indices (labels `y`).

        Parameters:
        -----------
        x: tf.constant. tf.int32. shape=(N,).
            Data samples / context word indices in the vocab.
        y: tf.constant. tf.int32. shape=(N,).
            Labels / target word indices in the vocab.
        batch_size: int.
            Number of samples to include in each mini-batch.
        epochs: int.
            Network should train this many epochs.
        print_every: int.
            How often (in MINI-BATCHES) should the network print progress and record the training loss?
        linear_lr_decay: bool.
            Do we apply a linear learning rate decay on every MINI-BATCH?
        linear_lr_min_lr: float.
            The minimum allowable learning rate if doing lr decay. We do not allow the lr to go below this value.
        verbose: bool.
            If set to `False`, there should be no print outs during training. Messages indicating start and end of
            training are fine.

        Returns:
        -----------
        train_loss_hist: Python list of floats.
            Training loss averaged over the most recent `print_every` MINI-BATCHES.
            For example: if `print_every`=5000, train_loss_hist looks like:
            [avg_loss(batches 0-4999), avg_loss(batches 5000-9999), ...]

        TODO:
        1. Use your DeepNetwork training method as a starting point, but this should be MUCH simpler :)
        You are essentially removing/simplying for current training loop here (e.g. remove val set support, early
        stopping, etc.)
        2. The only mechanism you should add is the linear learning rate decay (involving `lr_linear_decay` method)
        3. You will also need to revise how you are recording the training loss history. You `train_loss_hist` should
        contain training losses averaged over every `print_every` chunk of mini-batches.
        For example, if `print_every` is 500, then losses obtained from mini-batches 0-499 would be averaged and added
        as ONE entry in `train_loss_hist`, then the next 500 mini-batch losses (500-999) would be averaged then added as
        ONE entry in `train_loss_hist`, and so on.
        '''
        N = len(x)
        mini_batches = N // batch_size
        num_steps = epochs * mini_batches
        initial_lr = float(self.opt.learning_rate)
        
        train_loss_hist = []
        current_step = 0
        loss_accumulator = 0.0
        loss_count = 0
        
        # Ensure Dropout/Normalization are in training mode
        self.set_layer_training_mode(is_training=True)

        for e in range(epochs):
            # Shuffle at start of epoch to ensure variety in mini-batches
            indices = np.random.permutation(N)
            
            for b in range(mini_batches):
                # Sample mini-batch
                batch_idx = indices[b*batch_size : (b+1)*batch_size]
                batch_x = tf.gather(x, batch_idx)
                batch_y = tf.gather(y, batch_idx)
                
                # train_step is inherited from DeepNetwork
                loss = self.train_step(batch_x, batch_y)
                loss_accumulator += float(loss)
                loss_count += 1
                
                # Apply linear LR decay per batch
                if linear_lr_decay:
                    self.lr_linear_decay(initial_lr, current_step, num_steps, linear_lr_min_lr)
                
                # Record averaged loss history every print_every batches
                if loss_count % print_every == 0:
                    avg_loss = loss_accumulator / print_every
                    train_loss_hist.append(avg_loss)
                    
                    if verbose:
                        print(f'Epoch {e+1}/{epochs} | Step {current_step+1}/{num_steps} | '
                              f'Loss: {avg_loss:.4f} | LR: {float(self.opt.learning_rate):.6f}')
                    
                    loss_accumulator = 0.0
                    loss_count = 0
                
                current_step += 1
                
        if verbose: print(f'Finished training after {epochs} epochs!')
        return train_loss_hist

    def lr_linear_decay(self, initial_lr, t, num_steps, min_allowed_lr=1e-5):
        '''Applies a linear learning rate decay to the optimizer's learning rate on the MINI-BATCH level.

        See notebook for a refresher on the equation.

        Parameters:
        -----------
        initial_lr: float.
            The optimizer's lr at the BEGINNING of training, BEFORE any decay has taken place. This is constant over
            a training run.
        t: int.
            The current CUMULATIVE mini-batch number from the BEGINNING OF TRAINING (NOT the beginning of the epoch).
        num_steps: int.
            Total number of mini-batches that will be processed over the ENTIRETY of training (i.e. across ALL epochs).
        min_allowed_lr: float.
            We do not allow the linear lr decay to set the lr below this value.
            For example, if the lr decay equation says lr should be 0.001 but if min_allowed_lr=0.01, then we actually
            set the lr to 0.01.
        '''
        # Calculate decayed LR
        decayed_lr = initial_lr * (1.0 - (t + 1) / num_steps)
        # Ensure it doesn't dip below the minimum
        new_lr = max(decayed_lr, min_allowed_lr)
        # Update the optimizer's learning rate
        self.opt.learning_rate = new_lr

    def get_word_embedding(self, wordind):
        '''Given the word index `wordind` retrieve and return the corresponding embedding vector.'''
        return self.layers[0].wts[wordind]

    def get_all_embeddings(self):
        '''Retrieve and return the embedding vectors for ALL words in the vocab.'''
        return self.layers[0].wts

    def get_bias(self):
        '''Retrieve and return the embedding layer bias.'''
        return self.layers[0].bias

    def save_embeddings(self, path='export', filename='embeddings.npz'):
        '''Saves the embeddings to disk.

        This function is provided to you. You should not need to modify it.

        Parameters:
        -----------
        path: str.
            Folder path where the embeddings should be saved.
        filename: str.
            Name of the file to which the embeddings should be saved. Should have a .npz file extension.
        '''
        full_path = os.path.join(path, filename)

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        wts = self.get_all_embeddings()
        b = self.get_bias()
        np.savez_compressed(full_path, embeddings=wts, bias=b)
