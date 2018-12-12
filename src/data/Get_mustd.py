from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys
import numpy as np
import tensorflow as tf

from datetime import datetime


def read_tfrecord(tfrecord_file, train_position, batch_size = 1, num_epochs = None, shuffle = False, buffer_size = 99000):
    def parser(record):
        featnames = ['elements'] + [train_position] + ['targets']
        feattypes = [tf.int64, tf.float32, tf.float32]
        features = { featname :tf.FixedLenFeature([], tf.string) for featname in featnames}
        tfrecord_features = tf.parse_single_example(record, features=features, name='features')
        featlist = []
        for featname, feattype in zip(featnames, feattypes):
            feat = tf.decode_raw(tfrecord_features[featname], feattype)
            if featname == 'positions' or featname == 'mmffpositions' or featname == 'positions1' or featname == "positions2":
                feat = tf.reshape(feat, (-1, 3))
            if featname == 'targets':
                feat = tf.reshape(feat, (15, 1))
            featlist.append(feat)
        return featlist
    df = tf.data.TFRecordDataset(tfrecord_file)
    df = df.map(parser)
    if shuffle:
        df = df.shuffle(buffer_size = buffer_size)
    df = df.repeat(num_epochs)
    df = df.padded_batch(batch_size= batch_size,padded_shapes=([None,],[None,3],[15,1]),padding_values=None)
    iterator = df.make_initializable_iterator()
    features = iterator.get_next()
    in_names = ['elements'] + [train_position] + ['targets']
    features = dict(list(zip(in_names, features)))

    return iterator, features



def getmustd(traintf, train_position,reference,target, size, refidx = 1):
    """
    Input training TFrecord file and Atomic Energy Reference file
    Return: mu, std, atom_ref
    """
    reference = np.load(reference)
    if refidx != False:
        atom_ref = reference['atom_ref'][:, refidx:(refidx+1)]
    idx = 0
    E0n = []
    iter_train, features = read_tfrecord(traintf,train_position, num_epochs = 1, shuffle = False)
    with tf.Session() as sess:
        sess.run(iter_train.initializer)
        for n in range(size):
            feature_val = sess.run(features)

            z = feature_val["elements"][0]
            u0 = feature_val["targets"][0][target]
            if refidx != False:
                e0 = np.sum(atom_ref[z], 0)
                e0n = (u0-e0)/len(z)
            else:
                e0n = (u0)/len(z)
            E0n.append(e0n)
            idx += 1
            if idx % 100000 == 0:
                print(idx)

        mu = np.mean(E0n, axis = 0)
        std = np.std(E0n, axis = 0)
        return mu, std



def getmustd_final(infile,position,size,reference,outfile):
    for t in range(3,15):
        if t == 9:
            refidx = 0
        elif t == 10:
            refidx = 1
        elif t == 11:
            refidx = 2
        elif t == 12:
            refidx = 3
        elif t == 13:
            refidx = 4
        elif t == 14:
            refidx = 5
        else:
            refidx = False
        mu,std = getmustd(infile,position,reference, t, size, refidx)
        outfile.write(str(t) + "," + str(mu[0]) + "," + str(std[0]) + "\n")
    outfile.close()

    return None


if __name__ == "__main__":
    traintf = "../../data/processed/qm9mmff/train.tfrecord"
    size = 99000
    train_position = "positions"
    reference ="../../data/raw/atomrefs.txt.npz"
    outfile = open("../../data/processed/qm9mmff/reference.csv","w")
    getmustd_final(traintf,train_position,size, reference,outfile)
