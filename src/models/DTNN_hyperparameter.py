import os
import tensorflow as tf
import numpy as np
from DTNN_readtfrecord_mmff_cutoff_samestep import read_tfrecord
from DTNN_model_basis_reuse import Model as Model_reuse
from DTNN_model_basis_reusenone_activation import Model as Model_reusenone


def get_optimizer(opt,lr):
    """ Get Optimizer
    :param opt: optimizer type
    :param lr: learning rate
    """
    if opt == "Adam":
        optimizer = tf.train.AdamOptimizer(lr)
    elif opt == "GD":
        optimizer = tf.train.GradientDescentOptimizer(lr)

    return optimizer

def get_model(train_features,
                mu, std, atom_ref,
                reuse, n_basis, n_interactions, 
                activation):
    """ Get Model 
    
    :param reuse: [description]
    :type reuse: [type]
    :param train_features: [description]
    :type train_features: [type]
    :param n_basis: [description]
    :type n_basis: [type]
    :param n_interactions: [description]
    :type n_interactions: [type]
    :param activation: [description]
    :type activation: [type]
    :param mu: [description]
    :type mu: [type]
    :param std: [description]
    :type std: [type]
    :param atom_ref: [description]
    :type atom_ref: [type]
    :return: [description]
    :rtype: [type]
    """

    model = Model_reuse(train_features,n_basis,n_interactions, activation, mu, std, atom_ref) if reuse == "reuse" else Model_reusenone(train_features,n_basis,n_interactions, activation, mu, std, atom_ref)
    
    return model.mymodel()

def get_lr(learning, lr_int, i, global_step, cycle_steps):
    if i == 0:
        lr = lr_int
    else:
        lr = lr_int/2 * (np.cos(np.pi * np.mod(i-1,np.ceil(cycle_steps))/np.ceil(cycle_steps)) + 1)

    return lr


