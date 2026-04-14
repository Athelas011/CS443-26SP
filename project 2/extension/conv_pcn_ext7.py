"""Flexible ConvPCN architectures for Extension 7.

These classes are designed to sit next to the user's existing conv_pcn.py and let
her compare a baseline-style architecture against a paper-inspired variant with:
    - explicit bypass connections inside each PCN block
    - configurable pooling schedule
    - easy depth / width sweeps
"""

from __future__ import annotations

import network
from conv_layers import Conv2D, MaxPool2D
from layers import Dense, Dropout, Flatten
from conv_pcn_ext7_blocks import ConvPCNBlockBypass


class ConvPCNExt7(network.DeepNetwork):
    """Flexible ConvPCN architecture for architecture experiments.

    Structure:
        Input -> initial Conv2D -> N x modified ConvPCN block -> Flatten -> Dropout -> Dense -> Dense
    """

    def __init__(
        self,
        input_feats_shape,
        C,
        conv_units: int = 64,
        pcn_units: tuple[int, ...] = (64, 128, 256),
        num_steps: int = 5,
        step_lr: float = 0.5,
        maxpool_after_block: tuple[bool, ...] | None = None,
        dense_units: int = 128,
        dropout_rate: float | None = 0.2,
        wt_init: str = "he",
        do_group_norm: bool = True,
        error_activation: str = "relu",
        block_dropout_rate: float | None = None,
        initial_conv_kernel: int = 3,
    ):
        super().__init__(input_feats_shape=input_feats_shape)

        if maxpool_after_block is None:
            maxpool_after_block = tuple(False for _ in pcn_units)
        if len(maxpool_after_block) != len(pcn_units):
            raise ValueError("maxpool_after_block must match pcn_units length")

        self.layers = []
        prev = None

        # Initial plain convolution.
        init_conv = Conv2D(
            name="Conv2D_init",
            units=conv_units,
            kernel_size=initial_conv_kernel,
            strides=1,
            activation="relu",
            prev_layer_or_block=prev,
            wt_init=wt_init,
            do_group_norm=do_group_norm,
        )

        
        self.layers.append(init_conv)
        prev = init_conv

        # Modified paper-inspired PCN blocks.
        for i, units in enumerate(pcn_units):
            block = ConvPCNBlockBypass(
                blockname=f"PCNBlockExt7_{i}",
                units=units,
                kernel_size=3,
                strides=1,
                num_steps=num_steps,
                state_lr=step_lr,
                dropout_rate=block_dropout_rate,
                wt_init=wt_init,
                do_group_norm=do_group_norm,
                prev_layer_or_block=prev,
                error_activation=error_activation,
                feedback_activation="linear",
                bypass_activation="linear",
            )
            self.layers.append(block)
            prev = block

            if maxpool_after_block[i]:
                pool = MaxPool2D(
                    name=f"MaxPool_{i}",
                    pool_size=2,
                    strides=2,
                    prev_layer_or_block=prev,
                )
                self.layers.append(pool)
                prev = pool

        flatten = Flatten(name="Flatten", prev_layer_or_block=prev)
        self.layers.append(flatten)
        prev = flatten

        if dropout_rate is not None:
            drop = Dropout(name="Dropout", rate=dropout_rate, prev_layer_or_block=prev)
            self.layers.append(drop)
            prev = drop

        dense_hidden = Dense(
            name="Dense_hidden",
            units=dense_units,
            activation="relu",
            prev_layer_or_block=prev,
            wt_init=wt_init,
        )
        self.layers.append(dense_hidden)
        prev = dense_hidden

        output = Dense(
            name="Dense_output",
            units=C,
            activation="softmax",
            prev_layer_or_block=prev,
            wt_init=wt_init,
        )
        self.layers.append(output)
        self.output_layer = output

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


# -----------------------------
# Ready-to-use experiment builds
# -----------------------------

def build_ext7_small(input_shape, num_classes):
    """Fast architecture for quick debugging on a CIFAR subset."""
    return ConvPCNExt7(
        input_feats_shape=input_shape,
        C=num_classes,
        conv_units=32,
        pcn_units=(64, 96, 128),
        num_steps=1,
        step_lr=0.05,
        maxpool_after_block=(False, True, True),
        dense_units=96,
        dropout_rate=0.2,
        wt_init="he",
        do_group_norm=False,
    )


def build_ext7_medium(input_shape, num_classes):
    """Paper-inspired medium model with skip merge in every block."""
    return ConvPCNExt7(
        input_feats_shape=input_shape,
        C=num_classes,
        conv_units=64,
        pcn_units=(64, 128, 128, 192),
        num_steps=5,
        step_lr=0.05,
        maxpool_after_block=(False, False, True, True),
        dense_units=128,
        dropout_rate=0.2,
        wt_init="he",
        do_group_norm=False,
    )


def build_ext7_deeper(input_shape, num_classes):
    """Deeper architecture for the final comparison if runtime allows."""
    return ConvPCNExt7(
        input_feats_shape=input_shape,
        C=num_classes,
        conv_units=64,
        pcn_units=(64, 96, 128, 192, 192),
        num_steps=7,
        step_lr=0.05,
        maxpool_after_block=(False, True, False, True, True),
        dense_units=128,
        dropout_rate=0.25,
        wt_init="he",
        do_group_norm=False,
    )
