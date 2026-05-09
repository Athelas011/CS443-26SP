"""Modified ConvPCN blocks for Extension 7.

Main idea from Han et al. (NeurIPS 2018):
- keep local recurrent predictive-coding updates inside each block
- add an explicit bypass / skip pathway from block input to block output
- optionally regularize with dropout after recurrent refinement
"""

from __future__ import annotations

import tensorflow as tf

import block
from conv_layers import Conv2D, Conv2DTranspose
from layers import Dropout


class ConvPCNBlockBypass(block.Block):
    """ConvPCN block with an explicit 1x1 bypass path.

    Compared with the baseline ConvPCNBlock:
    1. Adds a 1x1 bypass conv from block input -> block output.
    2. Uses a linear feedback / reconstruction path by default.
    3. Keeps local recurrent predictive-coding updates.

    The rough update is:
        z0 = FFConv(x)
        repeat T times:
            x_hat = FBConv(z)
            err = ReLU(x - x_hat)
            z = z + lr * FFConv(err)
        out = z + BPConv(x)

    idea: the final block output should merge the refined state with a direct bypass projection of the lower-layer input.
    """

    def __init__(
        self,
        blockname: str,
        units: int,
        kernel_size: int | tuple[int, int] = 3,
        strides: int = 1,
        num_steps: int = 5,
        state_lr: float = 0.5,
        dropout_rate: float | None = None,
        wt_init: str = "he",
        do_group_norm: bool = True,
        prev_layer_or_block=None,
        error_activation: str = "relu",
        feedback_activation: str = "linear",
        bypass_activation: str = "linear",
    ):
        super().__init__(blockname, prev_layer_or_block)

        self.num_steps = num_steps
        self.state_lr = state_lr
        self.error_activation = error_activation

        # Feedforward state-estimation path.
        self.conv = Conv2D(
            name=blockname + "_conv",
            units=units,
            kernel_size=kernel_size,
            strides=strides,
            activation="linear",
            wt_init=wt_init,
            do_group_norm=do_group_norm,
            prev_layer_or_block=prev_layer_or_block,
        )


        # Top-down reconstruction path.
        self.conv_t = Conv2DTranspose(
            name=blockname + "_convT",
            kernel_size=kernel_size,
            strides=strides,
            activation=feedback_activation,
            wt_init=wt_init,
            do_group_norm=do_group_norm,
            prev_layer_or_block=self.conv,
        )

        # Direct bypass / skip projection.
        self.bypass = Conv2D(
            name=blockname + "_bypass",
            units=units,
            kernel_size=1,
            strides=strides,
            activation=bypass_activation,
            wt_init=wt_init,
            do_group_norm=do_group_norm,
            prev_layer_or_block=prev_layer_or_block,
        )

        self.layers = [self.conv, self.conv_t, self.bypass]

        if dropout_rate is not None:
            self.dropout = Dropout(
                name=blockname + "_dropout",
                rate=dropout_rate,
                prev_layer_or_block=self.conv,
            )
            self.layers.append(self.dropout)
        else:
            self.dropout = None

    def _activate_error(self, error: tf.Tensor) -> tf.Tensor:
        if self.error_activation == "relu":
            return tf.nn.relu(error)
        if self.error_activation == "leaky_relu":
            return tf.nn.leaky_relu(error, alpha=0.1)
        if self.error_activation == "tanh":
            return tf.nn.tanh(error)
        if self.error_activation == "linear":
            return error
        raise ValueError(f"Unsupported error_activation: {self.error_activation}")

    def __call__(self, x):
        z = tf.nn.relu(self.conv(x))
        units_prev = x.shape[-1]

        for _ in range(self.num_steps):
            x_hat = self.conv_t(z, units_prev)
            error = self._activate_error(x - x_hat)
            delta_z = self.conv(error)
            z = z + self.state_lr * delta_z
        # paper suggestion
        out = z + self.bypass(x)

        if self.dropout is not None:
            out = self.dropout(out)

        return out
