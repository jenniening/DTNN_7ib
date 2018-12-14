import numpy as np
import tensorflow as tf
from DTNN_atom import interatomic_distances
from DTNN_atom import site_rdf

####
# Read Tfrecord File
####

def read_tfrecord(tfrecord_file, train_position, batch_size = 100, num_epochs = None, shuffle = False, buffer_size = 99000,cutoff = 20, step=0.2):
    def parser(record):
        featnames = ['elements'] + [train_position] + ['targets']
        feattypes = [tf.int64, tf.float32, tf.float32]
        features = { featname :tf.FixedLenFeature([], tf.string) for featname in featnames}
        tfrecord_features = tf.parse_single_example(record, features=features, name='features')
        featlist = []
        for featname, feattype in zip(featnames, feattypes):
            feat = tf.decode_raw(tfrecord_features[featname], feattype)
            if featname == 'positions' or featname == 'mmffpositions' or featname == "positions1" or featname == "positions2":
                feat = tf.reshape(feat, (-1, 3))
            if featname == 'targets':
                feat = tf.reshape(feat, (15, 1))
            featlist.append(feat)
        return featlist
    df = tf.data.TFRecordDataset(tfrecord_file)
    df = df.map(parser)
    if shuffle != False:
        df = df.shuffle(buffer_size = buffer_size)
    df = df.repeat(num_epochs)
    df = df.padded_batch(batch_size= batch_size,padded_shapes=([None,],[None,3],[15,1]),padding_values=None)
    iterator = df.make_initializable_iterator()
    features = iterator.get_next()
    in_names = ['elements'] + [train_position] + ['targets']
    features = dict(list(zip(in_names, features)))

    distances = interatomic_distances(tf.cast(features[train_position], tf.float32))
    rdf = site_rdf(distances, cutoff=cutoff, step=step, width=1)

    features["rdf"] = rdf
    return iterator, features




