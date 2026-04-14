"""Helper utilities for Extension 7 experiments on CIFAR.

These helpers are intentionally lightweight and do not assume a very specific
training API beyond the course project's usual compile / fit / evaluate pattern.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass
class DatasetSplit:
    x_train: tf.Tensor
    y_train: tf.Tensor
    x_val: tf.Tensor
    y_val: tf.Tensor


@dataclass
class ExperimentResult:
    name: str
    runtime_sec: float
    epoch_used: float | None
    final_val_acc: float | None
    final_train_loss: float | None
    final_val_loss: float | None


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)



def make_cifar_subset(
    x: tf.Tensor,
    y: tf.Tensor,
    total_samples: int = 3000,
    val_fraction: float = 0.2,
    seed: int = 42,
    stratified: bool = True,
) -> DatasetSplit:
    """Create a fast train/val split from a local CIFAR dev set.
    """
    set_global_seed(seed)

    x_np = np.array(x)
    y_np = np.array(y)

    n = min(total_samples, len(x_np))

    if stratified:
        classes = np.unique(y_np)
        per_class = max(1, n // len(classes))
        idxs = []
        for c in classes:
            c_idx = np.where(y_np == c)[0]
            choose = min(per_class, len(c_idx))
            idxs.extend(np.random.choice(c_idx, size=choose, replace=False).tolist())
        idxs = np.array(idxs)
        if len(idxs) > n:
            idxs = np.random.choice(idxs, size=n, replace=False)
    else:
        idxs = np.random.choice(len(x_np), size=n, replace=False)

    np.random.shuffle(idxs)
    x_sub = x_np[idxs]
    y_sub = y_np[idxs]

    val_n = max(1, int(len(x_sub) * val_fraction))
    x_val = tf.constant(x_sub[:val_n], dtype=tf.float32)
    y_val = tf.constant(y_sub[:val_n], dtype=tf.int32)
    x_train = tf.constant(x_sub[val_n:], dtype=tf.float32)
    y_train = tf.constant(y_sub[val_n:], dtype=tf.int32)

    return DatasetSplit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)



def describe_split(split: DatasetSplit) -> None:
    print("x_train:", split.x_train.shape, "y_train:", split.y_train.shape)
    print("x_val:  ", split.x_val.shape, "y_val:  ", split.y_val.shape)



def default_experiment_configs():
    """
    1. small_fast: quick sanity check
    2. medium_paper: paper-inspired main model
    3. deeper_pooled: deeper / more pooled comparison
    """
    return [
        {
            "name": "small_fast",
            "builder": "build_ext7_small",
            "epochs": 5000,
            "batch_size": 32,
            "lr": 1e-3,
        },
        {
            "name": "medium_paper",
            "builder": "build_ext7_medium",
            "epochs": 5000,
            "batch_size": 32,
            "lr": 1e-3,
        },
        {
            "name": "deeper_pooled",
            "builder": "build_ext7_deeper",
            "epochs": 5000,
            "batch_size": 32,
            "lr": 8e-4,
        },
    ]



def run_with_project_api(model, split: DatasetSplit, epochs: int, batch_size: int, lr: float):
 
    # compile
    model.compile(loss="cross_entropy",lr=1e-3, print_summary=True)

    start = time.time()
    train_loss_hist, val_loss_hist, val_acc_hist, e = model.fit(
        split.x_train,
        split.y_train,
        x_val=split.x_val,
        y_val=split.y_val,
        batch_size=batch_size,
        max_epochs = epochs,
        print_every=10,
        patience = 7,
        lr_patience = 3,
        
    )
    runtime_sec = time.time() - start

    val_acc, val_loss = model.evaluate(split.x_val, split.y_val, batch_sz=batch_size)

    epoch_used = e
    final_train_loss = float(train_loss_hist[-1]) if len(train_loss_hist) > 0 else None
    final_val_loss = float(val_loss_hist[-1]) if len(val_loss_hist) > 0 else None
    final_val_acc = float(val_acc_hist[-1]) if len(val_acc_hist) > 0 else None
 
    history = [train_loss_hist, val_loss_hist, val_acc_hist, epoch_used]

    return history, ExperimentResult(
        name=model.__class__.__name__,
        runtime_sec=runtime_sec,
        epoch_used=epoch_used,
        final_val_acc=final_val_acc,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
    )
