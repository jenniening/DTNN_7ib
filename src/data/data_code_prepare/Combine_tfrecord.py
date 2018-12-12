import tensorflow as tf
import numpy as np
import tarfile

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

writer = tf.python_io.TFRecordWriter('db/split_5/train_new.tfrecord')

n_rows = 0
for record in tf.python_io.tf_record_iterator('db/split_5/train.tfrecord'):
    n_rows += 1
print(n_rows)

n_rows_1 = 0
for record in tf.python_io.tf_record_iterator('db/newmolecules/new_molecules_new.tfrecord'):
    n_rows_1 += 1
print(n_rows_1) 

   
featnames = ['elements', 'positions','mmffpositions', 'targets','extras']
    
def read_from_tfrecord(tfrecords_file, batch_size=1, isAtomType=False):
    
    tfrecord_file_queue = tf.train.string_input_producer(tfrecords_file, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)
    
    featnames = ['elements', 'positions','mmffpositions', 'targets','extras']
    feattypes = [tf.int64, tf.float32,tf.float32, tf.float32,tf.float32]
    if isAtomType:
        featname.append('atomtypes')
        feattypes.append(tf.int64)
    
    features = { featname :tf.FixedLenFeature([], tf.string) for featname in featnames}
        
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                        features=features, name='features')
    
    # list to save decoded features
    featlist = []
    for featname, feattype in zip(featnames, feattypes):
        feat = tf.decode_raw(tfrecord_features[featname], feattype)
        if featname == 'positions' or featname == 'mmffpositions':
            feat = tf.reshape(feat, (-1, 3))
        featlist.append(feat)
        
    return featlist

def read_tfrecord_iterator(tfrecord_file, n_rows):
    """
    This is a tfrecord iterator to go through each value in tfrecord
    """
    
    features = read_from_tfrecord([tfrecord_file])
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(n_rows):

            feature = sess.run(features)
            yield feature

        coord.request_stop()
        coord.join(threads)
    
# read feature in tf and write to each writer

for f in read_tfrecord_iterator('db/split_5/train.tfrecord', n_rows):
    features = {}
    for idx, featname in enumerate(featnames):
        feat = f[idx]
        if featname == 'positions' or featname == 'mmffpositions':
            feat = feat.ravel()
        features[featname] = _bytes_feature(tf.compat.as_bytes(feat.tostring()))

        
    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())
        
for f in read_tfrecord_iterator("db/newmolecules/new_molecules_new.tfrecord", n_rows_1):
    features = {}
    for idx, featname in enumerate(featnames):
        feat = f[idx]
        if featname == 'positions' or featname == "mmffpositions":
            feat = feat.ravel()
        features[featname] = _bytes_feature(tf.compat.as_bytes(feat.tostring()))

        
    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())
writer.close()
