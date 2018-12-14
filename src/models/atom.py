### Gaussian Expansion ###
import numpy as np
import tensorflow as tf

def interatomic_distances(positions):
    """
    Calculate interatomic distance.
    The input should be only one molecule each time.
    """
    with tf.variable_scope('distance'):
        positions = tf.expand_dims(positions, 2)
        rpos = tf.expand_dims(positions, 1)
        positions = tf.expand_dims(positions, 2)
        distances = tf.sqrt(
            tf.reduce_sum(tf.square(positions - rpos),
                          reduction_indices=4))

    return distances

def site_rdf(distances, cutoff, step, width, eps=1e-5,
             use_mean=False, lower_cutoff=None):
    """
    The input should be only one molecule each time.
    """
    with tf.variable_scope('srdf'):
        if lower_cutoff is None:
            vrange = cutoff
        else:
            vrange = cutoff - lower_cutoff
        distances = tf.expand_dims(distances, -1)
        n_centers = np.ceil(vrange / step)
        gap = vrange - n_centers * step
        n_centers = int(n_centers)

        if lower_cutoff is None:
            centers = tf.linspace(0., cutoff - gap, n_centers)
        else:
            centers = tf.linspace(lower_cutoff + 0.5 * gap, cutoff - 0.5 * gap,
                                  n_centers)
        centers = tf.reshape(centers, (1, 1, 1, -1))

        gamma = -0.5 / width / step ** 2

        rdf = tf.exp(gamma * (distances - centers) ** 2)

        mask = tf.cast(distances >= eps, tf.float32)
        rdf *= mask
        rdf = tf.reduce_sum(rdf, 3)
        if use_mean:
            N = tf.reduce_sum(mask, 2)
            N = tf.maximum(N, 1)
            rdf /= N

        new_shape = [None, None, None, n_centers]
        rdf.set_shape(new_shape)

    return rdf