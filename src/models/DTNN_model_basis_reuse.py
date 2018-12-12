# The Training Part of DTNN Model
import numpy as np
import tensorflow as tf

####
# Models Code
####

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

        new_dims = x_shape[:-1] + [n_out]
        y.set_shape(new_dims)
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

    def __init__(self, features,n_basis,n_interactions, activation, mu, std, atom_ref):
        self.features = features
        self.mu = mu
        self.std = std
        self.atom_ref = atom_ref
        self.n_basis = n_basis
        self.n_interactions = n_interactions

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

        # masking
        mask = tf.cast(tf.expand_dims(Z, 1) * tf.expand_dims(Z, 2),
                       tf.float32)
        diag = tf.matrix_diag_part(mask)
        diag = tf.ones_like(diag)
        offdiag = 1 - tf.matrix_diag(diag)
        mask *= offdiag
        mask = tf.expand_dims(mask, -1)

        I = np.eye(20).astype(np.float32)
        ZZ = tf.nn.embedding_lookup(I, Z)
        print(tf.shape(ZZ))
        r = tf.sqrt(1. / tf.sqrt(float(self.n_basis)))
        X = dense(ZZ, self.n_basis, use_bias=False,
                    weight_init=tf.random_normal_initializer(stddev=r))

        fC = dense(C, n_factors, use_bias=True)

        reuse = None
        for i in range(self.n_interactions):
            print(i)

            tmp = tf.expand_dims(X, 1)
            fX = dense(tmp, n_factors, use_bias=True,
                         scope='in2fac', reuse=reuse)

            fVj = fX * fC

            Vj = dense(fVj, self.n_basis, use_bias=False,
                         weight_init=tf.constant_initializer(0.0),
                         nonlinearity=tf.nn.tanh,
                         scope='fac2out', reuse=reuse)

            V = masked_sum(Vj, mask, axes=2)

            X += V
            reuse = True

            # output
        o1 = dense(X, self.n_basis // 2, nonlinearity=tf.nn.tanh)

        yi = dense(o1, 1,
                     weight_init=tf.constant_initializer(0.0),
                     use_bias=True)

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












