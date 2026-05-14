'''time_series_net.py
CNN-GRU and plain-GRU networks for multivariate time-series regression.
Ariel Pan, Daniel Lyu
CS 443: Bio-inspired Machine Learning
Project 4 Extension: CNN-GRU for Time Series Forecasting

Reference architecture: Sajjad et al., "A Novel CNN-GRU-Based Hybrid Approach
for Short-Term Residential Load Forecasting," IEEE Access 2020.
'''
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import tensorflow as tf

from layers import Dense, Dropout
from rnn_layers import GRU
from network import DeepNetwork
from conv1d_layer import Conv1D


class TimeSeriesNet(DeepNetwork):
    '''Base network for multivariate time-series regression.

    Key differences from DeepNetwork (classification):
    - Loss is MSE rather than cross-entropy.
    - Forward pass takes the *last* timestep from the final Dense output,
      so the network maps (B, T, n_in) → (B, n_out).
    - GRU layers receive an all-ones mask (no padding in time-series windows).
    - fit() tracks loss only (no accuracy).

    Reuses from DeepNetwork: compile(), summary(), get_all_params(),
    save_wts(), load_wts(), early_stopping(), lr_step_decay().
    '''

    def __init__(self, input_feats_shape, n_features_out):
        '''
        Parameters:
        -----------
        input_feats_shape: tuple. (T, n_in) — window length and feature count.
        n_features_out: int. Number of output features to predict.
        '''
        super().__init__(input_feats_shape)
        self.n_features_out = n_features_out
        self.is_recurrent_layer = []
        self._layers_forward = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_forward_layers(self):
        '''Walk the linked list tail→head and reverse to get input→output order.'''
        layers_fwd = []
        curr = self.output_layer
        while curr is not None:
            layers_fwd.append(curr)
            curr = curr.get_prev_layer_or_block()
        layers_fwd.reverse()
        return layers_fwd

    # ------------------------------------------------------------------
    # Override DeepNetwork methods
    # ------------------------------------------------------------------

    def compile(self, loss='mse', lr=1e-3, print_summary=True):
        '''Build forward layer list first, then delegate to parent compile.'''
        self._layers_forward = self._build_forward_layers()
        super().compile(loss=loss, lr=lr, print_summary=print_summary)

    def set_layer_training_mode(self, is_training):
        '''Set is_training tf.Variable in every layer.'''
        for layer in self._layers_forward:
            layer.set_mode(is_training)

    def __call__(self, x):
        '''Forward pass: CNN feature extraction → GRU temporal modeling → Dense.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, T, n_in).

        Returns:
        --------
        tf.float32 tensor. shape=(B, n_out).
            Prediction for the next timestep.
        '''
        curr_act = tf.cast(x, tf.float32)
        B = tf.shape(curr_act)[0]
        T = curr_act.shape[1]                              # static int at trace time
        mask = tf.ones([B, T, 1], dtype=tf.float32)       # no padding in TS windows

        for layer, is_rec in zip(self._layers_forward, self.is_recurrent_layer):
            if is_rec:
                curr_act = layer(curr_act, mask=mask)     # GRU: (B, T, H)
            else:
                curr_act = layer(curr_act)                # Conv1D / Dropout / Dense

        # Dense applied to (B, T, H) produces (B, T, n_out) via batched matmul;
        # take the last timestep to get the one-step-ahead prediction.
        return curr_act[:, -1, :]                         # (B, n_out)

    def loss(self, pred, y_true, mask=None):
        '''MSE regression loss. The mask parameter is accepted but unused —
        it exists so this signature is compatible with the parent's call sites.'''
        return tf.reduce_mean(
            tf.square(tf.cast(pred, tf.float32) - tf.cast(y_true, tf.float32))
        )

    @tf.function
    def train_step(self, x_batch, y_batch):
        '''One mini-batch forward + backward pass with gradient clipping.

        Parameters:
        -----------
        x_batch: tf.float32 tensor. shape=(B, T, n_in).
        y_batch: tf.float32 tensor. shape=(B, n_out).

        Returns:
        --------
        float. MSE loss for this batch.
        '''
        with tf.GradientTape() as tape:
            pred = self(x_batch)
            loss = self.loss(pred, y_batch)
        grads = tape.gradient(loss, self.all_net_params)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.opt.apply_gradients(zip(grads, self.all_net_params))
        return loss

    @tf.function
    def test_step(self, x_batch, y_batch):
        '''Validation forward pass.

        Returns (0.0, mse_loss) — the dummy 0.0 satisfies the parent's
        evaluate() interface which expects (acc, loss).
        '''
        pred = self(x_batch)
        loss = self.loss(pred, y_batch)
        return 0.0, loss

    # ------------------------------------------------------------------
    # Regression-specific evaluation
    # ------------------------------------------------------------------

    def evaluate_regression(self, x, y, batch_sz=64):
        '''Compute MSE, MAE, RMSE and return all predictions.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(N, T, n_in).
        y: tf.float32 tensor. shape=(N, n_out).
        batch_sz: int. Inference mini-batch size.

        Returns:
        --------
        mse, mae, rmse: float.
        preds: np.ndarray. shape=(N, n_out).
        '''
        self.set_layer_training_mode(False)
        N = len(x)
        num_batches = max(1, N // batch_sz)
        all_preds = []
        for b in range(num_batches):
            xb = x[b * batch_sz:(b + 1) * batch_sz]
            all_preds.append(self(xb).numpy())
        self.set_layer_training_mode(True)

        preds = np.concatenate(all_preds, axis=0)
        y_np = y[:len(preds)].numpy() if hasattr(y, 'numpy') else np.array(y[:len(preds)])
        mse  = float(np.mean((preds - y_np) ** 2))
        mae  = float(np.mean(np.abs(preds - y_np)))
        rmse = float(np.sqrt(mse))
        return mse, mae, rmse, preds

    # ------------------------------------------------------------------
    # Regression training loop
    # ------------------------------------------------------------------

    def fit(self, x, y, x_val=None, y_val=None, batch_size=32, max_epochs=50,
            val_every=1, verbose=True, patience=10, lr_patience=5,
            lr_decay_factor=0.5, lr_max_decays=5):
        '''Train the network for regression.

        Parameters:
        -----------
        x, y: training inputs and targets (tf.Tensor or np.ndarray).
        x_val, y_val: optional validation split.
        batch_size: int. Mini-batch size.
        max_epochs: int. Maximum training epochs.
        val_every: int. Validate every this many epochs.
        verbose: bool. Print progress.
        patience: int. Early-stopping window (validation loss epochs).
        lr_patience: int. LR-decay window (validation loss epochs).
        lr_decay_factor: float. LR multiplier on decay.
        lr_max_decays: int. Max number of LR decays allowed.

        Returns:
        --------
        train_loss_hist: list of float. Per-epoch mean training MSE.
        val_loss_hist: list of float. Per-validation-check MSE.
        e: int. Epochs actually run.
        '''
        train_loss_hist, val_loss_hist = [], []
        N = len(x)
        num_batches = max(1, N // batch_size)
        rng = np.random.default_rng(0)
        self.set_layer_training_mode(True)
        recent_losses, recent_lr_losses, lr_decays, e = [], [], 0, 0

        for epoch in range(max_epochs):
            e = epoch + 1
            epoch_loss = 0.0

            for _ in range(num_batches):
                idx = rng.integers(0, N, batch_size)
                batch_loss = self.train_step(
                    tf.gather(x, idx), tf.gather(y, idx)
                )
                epoch_loss += float(batch_loss)

            epoch_loss /= num_batches
            train_loss_hist.append(epoch_loss)

            if e % val_every == 0 and x_val is not None:
                self.set_layer_training_mode(False)
                mse, mae, rmse, _ = self.evaluate_regression(x_val, y_val)
                self.set_layer_training_mode(True)
                val_loss_hist.append(mse)

                if verbose:
                    print(f'Epoch {e:>4}/{max_epochs} | '
                          f'train_MSE: {epoch_loss:.6f} | '
                          f'val_MSE: {mse:.6f} | val_RMSE: {rmse:.6f}')

                recent_losses, stop = self.early_stopping(recent_losses, mse, patience)
                if stop:
                    if verbose:
                        print('Early stopping triggered.')
                    break

                if lr_decays < lr_max_decays:
                    recent_lr_losses, decay = self.early_stopping(
                        recent_lr_losses, mse, lr_patience
                    )
                    if decay:
                        self.lr_step_decay(lr_decay_factor)
                        lr_decays += 1

            elif verbose:
                print(f'Epoch {e:>4}/{max_epochs} | train_MSE: {epoch_loss:.6f}')

        return train_loss_hist, val_loss_hist, e


# ======================================================================
# Concrete network architectures
# ======================================================================

class GRU_Regressor(TimeSeriesNet):
    '''Baseline plain-GRU model for multivariate time-series forecasting.

    Architecture:
        Input (B, T, n_in)
        → GRU(gru_units[0])    shape: (B, T, H0)
        → GRU(gru_units[1])    shape: (B, T, H1)
        → Dense(n_out, linear) shape: (B, T, n_out)
        → last timestep        shape: (B, n_out)
    '''

    def __init__(self, input_feats_shape, n_features_out, gru_units=(64, 32)):
        '''
        Parameters:
        -----------
        input_feats_shape: tuple. (T, n_in).
        n_features_out: int. Number of output features.
        gru_units: tuple of int. Hidden units in each GRU layer.
        '''
        super().__init__(input_feats_shape, n_features_out)

        self.gru1  = GRU('gru1',   gru_units[0], prev_layer_or_block=None)
        self.gru2  = GRU('gru2',   gru_units[1], prev_layer_or_block=self.gru1)
        self.dense = Dense('output', n_features_out, activation='linear',
                           prev_layer_or_block=self.gru2)

        self.is_recurrent_layer = [True, True, False]
        self.output_layer = self.dense


class CNN_GRU_Regressor(TimeSeriesNet):
    '''CNN-GRU hybrid model for multivariate time-series forecasting.

    Implements the architecture from Sajjad et al. (IEEE Access, 2020):
    two Conv1D layers extract local temporal features (playing the role of
    "spatial feature extraction" in the original electricity domain), then
    two GRU layers model long-range temporal dependencies, and a Dense layer
    produces the one-step-ahead prediction.

    Architecture:
        Input (B, T, n_in)
        → Conv1D(cnn_filters[0], k=kernel_size, relu)   (B, T, F0)
        → Dropout(dropout_rates[0])
        → Conv1D(cnn_filters[1], k=kernel_size, relu)   (B, T, F1)
        → Dropout(dropout_rates[1])
        → GRU(gru_units[0])                             (B, T, H0)
        → GRU(gru_units[1])                             (B, T, H1)
        → Dense(n_out, linear)                          (B, T, n_out)
        → last timestep                                 (B, n_out)
    '''

    def __init__(self, input_feats_shape, n_features_out,
                 cnn_filters=(16, 8), kernel_size=2,
                 dropout_rates=(0.1, 0.1), gru_units=(64, 32)):
        '''
        Parameters:
        -----------
        input_feats_shape: tuple. (T, n_in).
        n_features_out: int. Number of output features.
        cnn_filters: tuple of int. Filters in each Conv1D layer.
        kernel_size: int. Kernel length for both Conv1D layers.
        dropout_rates: tuple of float. Dropout rates after each Conv1D.
        gru_units: tuple of int. Hidden units in each GRU layer.
        '''
        super().__init__(input_feats_shape, n_features_out)

        self.conv1 = Conv1D('conv1', cnn_filters[0], kernel_size=kernel_size,
                            activation='relu', prev_layer_or_block=None, wt_init='he')
        self.drop1 = Dropout('drop1', rate=dropout_rates[0],
                             prev_layer_or_block=self.conv1)
        self.conv2 = Conv1D('conv2', cnn_filters[1], kernel_size=kernel_size,
                            activation='relu', prev_layer_or_block=self.drop1, wt_init='he')
        self.drop2 = Dropout('drop2', rate=dropout_rates[1],
                             prev_layer_or_block=self.conv2)
        self.gru1  = GRU('gru1',   gru_units[0], prev_layer_or_block=self.drop2)
        self.gru2  = GRU('gru2',   gru_units[1], prev_layer_or_block=self.gru1)
        self.dense = Dense('output', n_features_out, activation='linear',
                           prev_layer_or_block=self.gru2)

        self.is_recurrent_layer = [False, False, False, False, True, True, False]
        self.output_layer = self.dense
