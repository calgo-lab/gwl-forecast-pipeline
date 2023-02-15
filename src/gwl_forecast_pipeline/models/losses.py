def mean_group_mse(y_true, y_pred):
    """
    calculate mean squared error for each group separately and return the average
    last column corresponds to group index, mean squared error calculated over all other columns

    Args:
        y_true: 2-dim array, last column corresponds to group index
        y_pred: 2-dim array, last column corresponds to group index

    Returns:
        tf.Tensor: mean grouped mean squared error
    """
    import tensorflow as tf
    #import tensorflow_probability as tfp

    groups = tf.cast(y_true[:, -1], tf.int32)
    y_true, y_pred = y_true[:, :-1], y_pred[:, :-1]

    square = tf.math.square(y_pred - y_true)
    unique, idx, count = tf.unique_with_counts(groups)
    group_losses = tf.math.unsorted_segment_mean(square, idx, tf.size(unique))
    group_losses = tf.math.reduce_mean(group_losses, axis=1)
    mean = tf.math.reduce_mean(group_losses)
    return mean


def weighted_mse(y_true, y_pred):
    """
    calculate the weighted mean squared error

    Args:
        y_true: 2-dim array, last column corresponds to sample weights
        y_pred: 1-dim array

    Returns:
        tf.Tensor: weighted mean squared error
    """
    import tensorflow as tf
    weights = y_true[:, -1]
    y_true = y_true[:, :-1]
    sum_weights = tf.reduce_sum(weights)
    mse = tf.reduce_sum(weights * tf.square(y_true - y_pred))
    return mse / sum_weights

