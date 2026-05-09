'''image_datasets.py
Functions to load and preprocess image datasets
Ariel Pan, Daniel Lyu
CS 443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning
'''
import tensorflow as tf
import numpy as np

def get_dataset(name, norm_method='global', flatten=True, eps=1e-10, verbose=True):
    '''Main function to load the dataset `name` then preprocess and return it.

    Parameters:
    -----------
    name: str.
        Name of the requested dataset to retrieve. Supported options: 'mnist', 'cifar10'.
    norm_method: str.
        Method used to preprocess the images. Supported options: 'global', 'center', 'none'.
            - 'global' means images should be standardized using the mean and standard deviation RGB triplet computed
            globally across all images in the training set.
            - 'center' means images should be centered using the mean RGB triplet computed globally across all images in
            the training set.
    flatten: bool.
        Should we flatten out the non-batch dimensions so that the shape becomes (N, M)?
    eps: float.
        Small fudge factor to prevent potential division by 0 when standardizing.
    verbose: bool.
        When true, it is ok to print out shapes, dtypes, and other debug info. When false, nothing should print in this
        function.

    Returns:
    --------
    x_train: tf.float32 tensor. shape=(N_train, Iy, Ix, n_chans) or (N_train, M)
        Training set images
    y_train: tf.int32 tensor. shape=(N_train,)
        Training set labels
    x_test: tf tensor. shape=(N_test, Iy, Ix, n_chans) or (N_test, M)
        Test set images
    y_test: tf.int32 tensor. shape=(N_test,)
        Test set labels

    NOTE:
    1. You should rely on the TensorFlow Keras built-in datasets module to acquire the datasets.
    2. Work in NumPy to min-max normalize the images features to floats between 0-1 and then perform any additional
    preprocessing.
    3. Remember to cast the data to TensorFlow tensors of the appropriate data types before returning.
    4. When normalizing, you want to use the stats from the training set to normalize the test set
    (otherwise the exact same feature values would get mapped differently between the train and test sets, which is not
    good).
    '''
    if name != 'mnist' and name != 'cifar10':
        raise ValueError(f'Unsupported dataset name {name}. Supported options: "mnist", "cifar10".')
    if norm_method != 'global' and norm_method != 'center' and norm_method != 'none':
        raise ValueError(f'Unsupported norm_method {norm_method}. Supported options: "global", "center", "none".')
    
    ds = tf.keras.datasets.mnist if name == 'mnist' else tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = ds.load_data()
    # Min-max normalize to 0-1
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    # Compute global mean and std from training set for normalization
    if norm_method == 'global' or norm_method == 'center':
        mean_rgb = np.mean(x_train, axis=(0, 1, 2))
        if norm_method == 'global':
            std_rgb = np.std(x_train, axis=(0, 1, 2))
    # Normalize using global stats
    if norm_method == 'global':
        x_train = (x_train - mean_rgb) / (std_rgb + eps)
        x_test = (x_test - mean_rgb) / (std_rgb + eps)
    elif norm_method == 'center':
        x_train = x_train - mean_rgb
        x_test = x_test - mean_rgb
    # Flatten if specified
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    # Cast to tf tensors
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train.squeeze(), dtype=tf.int32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test.squeeze(), dtype=tf.int32)
    if verbose:
        print(f'x_train shape: {x_train.shape}, dtype: {x_train.dtype}')
        print(f'y_train shape: {y_train.shape}, dtype: {y_train.dtype}')
        print(f'x_test shape: {x_test.shape}, dtype: {x_test.dtype}')
        print(f'y_test shape: {y_test.shape}, dtype: {y_test.dtype}')
    return x_train, y_train, x_test, y_test


def train_val_split(x_train, y_train, prop_val=0.1):
    '''Subdivides the provided training set into a (smaller) training set and a validation set, composed of the last
    `prop_val` proportion of samples in x_train/y_train.

    Parameters:
    -----------
    x_train: tf.float32 tensor. shape=(N, Iy, Ix, n_chans) or (N, M)
        The original training set data
    y_train: tf.int32 tensor. shape=(N,)
        The original training set labels
    prop_val: float.
        Proportion of the original training set to reserve for the validation set.


    Returns:
    --------
    x_train: tf.float32 tensor. shape=(N_train_new, Iy, Ix, n_chans) or (N_train_new, M)
        Training set images
    y_train: tf.int32 tensor. shape=(N_train_new,)
        Training set labels
    x_val: tf tensor. shape=(N_val, Iy, Ix, n_chans) or (N_val, M)
        Validation set images
    y_val: tf.int32 tensor. shape=(N_val,)
        Validation set labels
    '''
    N = x_train.shape[0]
    N_val = int(N * prop_val)
    x_val = x_train[-N_val:]
    y_val = y_train[-N_val:]
    x_train_new = x_train[:-N_val]
    y_train_new = y_train[:-N_val]
    return x_train_new, y_train_new, x_val, y_val


def preprocess_nonlinear(x, n=4.0):
    '''Preprocessor for the nonlinear decoder input data. Applies the ReLU function raised to the `n` power.

    (Week 3)

    Parameters:
    -----------
    x: tf.float32 tensor. shape=(N, M)
        Hebbian network net_in values.
    n: float.
        Power to raise the output of the ReLU applied to `x`.

    Returns:
    --------
    tf.float32 tensor. shape=(N, M).
        Data transformed by ReLU raised to the `n` power.
    '''
    return tf.pow(tf.nn.relu(x), n)


def occlude_images(x, region='top', image_dims=(28, 28, 1)):
    '''Occludes/deletes the content in half of each image passed in.

    (This function is provided / should not required modification)

    Parameters:
    -----------
    x: tf.float32 tensor. shape=(N, M)
        Flatten image data.
    region: str.
        Region in each image to occlude. Supported options: 'top', 'bottom'
            - 'top' means occlude top half of images
            - 'bottom' means occlude bottom half of images
    image_dims: tuple of ints.
        The original unflattened shape of the image data without the batch dimension.

    Returns:
    --------
    x_flat: tf.float32 tensor. shape=(N, M).
        Occluded images.
    mask_flat: tf.float32 tensor. shape=(N, M).
        Boolean mask specifying whether occlusion was applied to each pixel each image.
    '''
    N, M = x.shape

    # Reshape to 2D
    x_2d = tf.reshape(x, [N, image_dims[0], image_dims[1], image_dims[2]])

    half_ind = image_dims[0] // 2

    if region == 'top':
        # Make mask to 0 out the top half of each image
        occlusion_mask = tf.concat([tf.zeros([N, half_ind, image_dims[1], image_dims[2]]),
                                    tf.ones([N, half_ind, image_dims[1], image_dims[2]])], axis=1)
    elif region == 'bottom':
        # Make mask to 0 out the bottom half of each image
        occlusion_mask = tf.concat([tf.ones([N, half_ind, image_dims[1], image_dims[2]]),
                                    tf.zeros([N, half_ind, image_dims[1], image_dims[2]])], axis=1)

    # Apply the mask: Where mask = 1, keep pixels. Where 0.0, set to min img value
    x_min = tf.reduce_min(x)
    x_occluded = tf.where(tf.cast(occlusion_mask, tf.bool), x_2d, x_min)

    # Flatten occluded images and mask back to (N, M)
    x_flat = tf.reshape(x_occluded, [N, M])
    mask_flat = tf.reshape(occlusion_mask, [N, M])
    return x_flat, mask_flat
