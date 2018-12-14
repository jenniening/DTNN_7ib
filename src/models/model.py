### Model Architecture ###
### Detailed description can be found in notebooks tutorial ### 

import numpy as np
import tensorflow as tf

def shape(x):
    if isinstance(x, tf.Tensor):
        return x.get_shape().as_list()
    return np.shape(x)

def glorot_uniform(shape, dtype, partition_info=None):
    if not dtype.is_floating:
        raise ValueError("Expected floating point type, got %s." % dtype)

    n_in = np.prod(shape[:-1])
    n_out = shape[-1]

    r = tf.cast(tf.sqrt(6. / (n_in + n_out)), tf.float32)
    return tf.random_uniform(shape, -r, r, dtype=dtype)

def reference_initializer(ref):
    def initializer(shape, dtype, partition_info=None):
        return tf.cast(tf.constant(np.reshape(ref, shape)), dtype)
    return initializer

def _softplus(x):
    return tf.log1p(tf.exp(x))

def shifted_softplus(x):
    y = tf.where(x < 14., _softplus(tf.where(x < 14., x, tf.zeros_like(x))), x)
    return y - tf.log(2.)


def dense(x, n_out,
          nonlinearity=None,
          use_bias=True,
          weight_init=glorot_uniform,
          bias_init=tf.constant_initializer(0.),
          trainable=True,
          scope=None, reuse=False, name='Dense'):
    x_shape = shape(x)
    ndims = len(x_shape)
    n_in = x_shape[-1]
    with tf.variable_scope(scope, default_name=name, values=[x],
                           reuse=reuse) as scope:
        # reshape for broadcasting
        xr = tf.reshape(x, (-1, n_in))

        W = tf.get_variable('W', shape=(n_in, n_out),
                            initializer=weight_init,
                            trainable=trainable)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
        tf.summary.histogram('W', W)

        y = tf.matmul(xr, W)

        if use_bias:
            b = tf.get_variable('b', shape=(n_out,),
                                initializer=bias_init,
                                trainable=trainable)
            tf.add_to_collection(tf.GraphKeys.BIASES, b)
            tf.summary.histogram('b', b)
            y += b

        if nonlinearity:
            y = nonlinearity(y)

        new_shape = tf.concat([tf.shape(x)[:ndims - 1], [n_out]], axis=0)
        y = tf.reshape(y, new_shape)

        #new_dims = x_shape[:-1] + [n_out]
        #y.set_shape(new_dims)
        tf.summary.histogram('activations', y)

    return y

def embedding(indices, n_vocabulary, n_out,
              weight_init=glorot_uniform,
              reference=None,
              trainable=True,
              scope=None, reuse=False, name='Embedding'):
    if type(n_out) is int:
        n_out = (n_out,)
    with tf.variable_scope(scope, default_name=name, reuse=reuse) as scope:
        if reference is None:
            W = tf.get_variable('W', shape=(n_vocabulary,) + n_out,
                                initializer=weight_init,
                                trainable=trainable)
        else:
            W = tf.get_variable('W', shape=(n_vocabulary,) + n_out,
                                initializer=reference_initializer(reference),
                                trainable=trainable)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)

        y = tf.nn.embedding_lookup(W, indices)
    return y

def masked_reduce(x, mask=None, axes=None,
                  reduce_op=tf.reduce_sum,
                  keep_dims=False,
                  scope=None, name='masked_reduce'):
    scope_vars = [x]
    if mask is not None:
        scope_vars.append(mask)

    with tf.variable_scope(scope, default_name=name,
                           values=scope_vars) as scope:
        if mask is not None:
            mask = tf.cast(mask > 0, tf.float32)
            x *= mask

        y = reduce_op(x, axes, keep_dims)

    return y

def masked_sum(x, mask=None, axes=None,
               keep_dims=False,
               scope=None, name='masked_sum'):
    return masked_reduce(x, mask, axes, tf.reduce_sum,
                         keep_dims, scope, name)


def masked_mean(x, mask=None, axes=None,
                keep_dims=False,
                scope=None, name='masked_mean'):
    if mask is None:
        mred = masked_reduce(x, mask, axes, tf.reduce_mean,
                             keep_dims, scope, name)
    else:
        msum = masked_reduce(x, mask, axes, tf.reduce_sum,
                             keep_dims, scope, name)
        mask = tf.cast(mask > 0, tf.float32)
        N = tf.reduce_sum(mask, axes, keep_dims)
        N = tf.maximum(N, 1)
        mred = msum / N
    return mred

class Model:

    def __init__(self, features, n_basis, n_interactions, activation,mu, std, atom_ref,num_readout_add,transferlearning):
        self.features = features
        self.mu = mu
        self.std = std
        self.atom_ref = atom_ref
        self.n_basis = n_basis
        self.n_interactions = n_interactions
        self.activation = activation
        self.transferlearning = transferlearning
        self.nr = num_readout_add

    def mymodel(self):

        """
        The features are in batching.
        """

        n_factors = 60

        mu = tf.constant(self.mu, tf.float32)
        std = tf.constant(self.std, tf.float32)
        atom_ref = tf.constant(self.atom_ref, tf.float32)

        Z = self.features['elements']
        C = self.features['rdf']

        ### masking ###

        mask = tf.cast(tf.expand_dims(Z, 1) * tf.expand_dims(Z, 2), tf.float32)
        mask = tf.matrix_set_diag(mask, tf.zeros_like(tf.matrix_diag_part(mask)))
        mask = tf.expand_dims(mask, -1)
        
        ### input initilization ###
        I = np.eye(20).astype(np.float32)
        ZZ = tf.nn.embedding_lookup(I, Z)
        r = tf.sqrt(1. / tf.sqrt(float(self.n_basis)))
        X = dense(ZZ, self.n_basis, use_bias=False,
                    weight_init=tf.random_normal_initializer(stddev=r))
        fC = dense(C, n_factors, use_bias=True)

        ### interaction block ###
        reuse = None
        for i in range(self.n_interactions):

            tmp = tf.expand_dims(X, 1)
            fX = dense(tmp, n_factors, use_bias=True,
                         scope='in2fac' + str(i), reuse=reuse)

            fVj = fX * fC
            
            if self.activation == "SSoftp":
                Vj = dense(fVj, self.n_basis, use_bias=False,
                         weight_init=tf.constant_initializer(0.0),
                         scope='fac2out' + str(i), reuse=reuse)
                Vj = shifted_softplus(Vj)
            elif self.activation == "selu":
                Vj = dense(fVj, self.n_basis, use_bias=False,
                         weight_init=tf.random_normal_initializer(stddev = 1.0/n_factors),
                         nonlinearity=tf.nn.selu,
                         scope='fac2out' + str(i), reuse=reuse)
                #Vj = tf.contrib.nn.alpha_droupout(Vj, 0.9)
            elif self.activation == "relu":
                Vj = dense(fVj, self.n_basis, use_bias=False,
                         weight_init=tf.constant_initializer(0.0),
                         nonlinearity=tf.nn.relu,
                         scope='fac2out' + str(i), reuse=reuse)
            else:
                Vj = dense(fVj, self.n_basis, use_bias=False,
                         weight_init=tf.constant_initializer(0.0),
                         nonlinearity=tf.nn.tanh,
                         scope='fac2out' + str(i), reuse=reuse)

            V = masked_sum(Vj, mask, axes=2)
            ### atom embedding update ###
            X += V
            reuse = None

        ### readout layers ###
        if self.transferlearning:
            o1 = dense(X, self.n_basis // 2)
            o1 = shifted_softplus(o1)
            o = o1
            num = 2
        else:
            o1 = dense(X, self.n_basis // 2, nonlinearity=tf.nn.tanh)
            o = o1
            num = 2
        ### add readout layers ###
        for i in range(self.nr):
            num = num * 2
            o = dense(o, self.n_basis//num, name = "new_" * (i + 1) + "layer")
            o = shifted_softplus(o)


        yi = dense(o, 1, weight_init=tf.constant_initializer(0.0), use_bias=True)

        #mu = tf.get_variable('mu', shape=(1,),
        #                         initializer=L.reference_initializer(mu),
        #                         trainable=False)
         #std = tf.get_variable('std', shape=(1,),
        #                          initializer=L.reference_initializer(std),
        #                          trainable=False)
        yi = yi * std + mu

        if atom_ref != 0.000:
            #E0i = embedding(Z, 100, 1, reference=atom_ref, trainable=False)
            E0i = tf.nn.embedding_lookup(atom_ref, Z)
            yi += E0i

        atom_mask = tf.expand_dims(Z, -1)
        mask = tf.cast(atom_mask > 0, tf.float32)

        y = tf.reduce_sum(yi * mask, 1)


        return {'y': y, 'y_i': yi, 'o1':o1, 'X':X}












