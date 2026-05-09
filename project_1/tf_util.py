'''tf_util.py
A few helper functions provided to you for various projects in CS 443.
Oliver W. Layton
CS 443: Bio-Inspired Machine Learning
'''
import tensorflow as tf


def arange_index(x, y):
    '''Reproduces arange indexing from NumPy in TensorFlow. I.e. Pick out the values in the column
    indices `y` as you go down the rows.

    Parameters:
    -----------
    x: tf.float32 tensor. shape=(B, C).
        A 2D tensor that we want to index with arange indexing.
    y: tf.float32 tensor. shape=(B,).
        The column indices to pick out from each row of `x`

    Returns:
    --------
    tf.float32 tensor. shape=(B,).
        Values from `x` extract from columns specified by `y`.

    Example:
    --------
    x = [[1., 2., 3.],
         [4., 5., 6.],
         [7., 8., 9.],
         [3., 2., 1.]]
    y = [1, 0, 2, 1]
    returns: [2., 4., 9., 2.]
    '''
    rows = tf.range(len(x))
    rc_tuples = tf.stack([rows, y], axis=1)
    return tf.gather_nd(x, rc_tuples)
