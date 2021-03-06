### This script is for preparing training, validtaion, test sets from QM9_MMFF dataset ###

### packages ###
import numpy as np
import tarfile
import tempfile
from tempfile import TemporaryDirectory
from ase.units import Hartree, eV, Bohr, Ang
import os
import tensorflow as tf

### functions ###

def split_data(partition,total_num, random_seed = 1, save = True):
    ''' This is function used to split data
    
    :param partition: provides size of training, validation, test_live
    :type partition: dic
    :param total_num: size of whole dataset
    :type total_num: int
    :param random_seed: defaults to 1
    :param random_seed: int, optional
    :param save: defaults to True
    :param save: bool, optional
    :return: index list of traing, validation, test_live and test
    :rtype: list
    '''

    ### total data index ###
    data_id_list = [i for i in range(1,total_num + 1)]
    ### get partion for each type of data ###
    [tran,valn,testln] = [partition[i] for i in ["train","validation","test_live"]]
    ### whether or not to use random seed
    if random_seed:
        np.random.seed(random_seed)
    np.random.shuffle(data_id_list)
    train_id = data_id_list[:tran]
    val_id = data_id_list[tran:tran + valn]
    test_live_id = data_id_list[tran + valn:tran + valn + testln]
    test_id = data_id_list[tran + valn + testln:]
    if save:
        np.savez("split_infor.npz",train= train_id, test = test_id, validation = val_id, test_live = test_live_id)
    
    return train_id, val_id, test_live_id, test_id

### Three functions to hold different types of input features ### 
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = value))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

def _float64_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def write_tfrecord_fromtar(tar, outfile,data_list):
    ''' This is function used to write data into tfrecord file format
    
    :param tar: initial tar file of dataset 
    :param outfile: tfrecord file 
    :type outfile: tfrecord
    :param data_list: index list 
    :type data_list: list
    '''
    element_conversions = {"H":1,"C":6,"O":8,"N":7,"F":9}
    olddir = os.getcwd()
    writer = tf.python_io.TFRecordWriter(outfile)
    for data in data_list:
        ### save into temporary directory which will be deleted automatically after exit ###
        with TemporaryDirectory() as temp_dir:
            #temp_dir = "../../data/processed/"
            #raw_path = temp_dir
            raw_path = os.path.join(temp_dir, 'qm9_mmff_file')
            #print(raw_path)
            tmpsdf ="./" + str(data) + ".sdf"
            tmpmmffsdf = "./" + str(data) + ".mmff.sdf"
            tar.extract(tmpsdf, path = raw_path)
            tar.extract(tmpmmffsdf, path = raw_path)
            #lines = str(tar.extractfile(tmpsdf).read(),'utf-8').split('\n')
            #linesm = str(tar.extractfile(tmpmmffsdf).read(),'utf-8').split('\n')
            os.chdir(raw_path)
            lines = open(tmpsdf,"r").readlines()
            linesm = open(tmpmmffsdf,"r").readlines()
            targets = [float(i) for i in lines[0].split()]
            ### using atom_num to get element number and coordinates ####
            atom_num = int(lines[3].split()[0])
            #print(atom_num)
            elements= [element_conversions[line.split()[3]] for line in lines[4:4+atom_num]]      
            positions = [line.split()[0:3] for line in lines[4:4+atom_num]]
            mmffpositions = [line.split()[0:3] for line in linesm[4:4+atom_num]]
            
            elements = np.array(elements).astype(np.int64)
            targets = np.array(targets).astype(np.float32)
            positions = np.array(positions).astype(np.float32)
            mmffpositions = np.array(mmffpositions).astype(np.float32)
            
            newfeatures = {'elements' : _bytes_feature([elements.tostring()]),
                           'positions' : _bytes_feature([positions.ravel().tostring()]),
                           'mmffpositions' : _bytes_feature([mmffpositions.ravel().tostring()]),
                           'targets' : _bytes_feature([targets.tostring()])}
            
            example = tf.train.Example(features=tf.train.Features(feature=newfeatures))
            writer.write(example.SerializeToString())
            os.chdir(olddir)
    writer.close()

### Write tfrecord from total tfrecord file ###
### This is much faster than writing tfrecord file from tar file ###

def __parse_function(record):
    features = {"elements":tf.FixedLenFeature([], tf.string),
                "positions":tf.FixedLenFeature([], tf.string),
                "mmffpositions":tf.FixedLenFeature([], tf.string),
                "targets":tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(record,features)
    features_new = {}
    dtype = {"elements":tf.int64,"targets":tf.float32,"positions":tf.float32,"mmffpositions":tf.float32}
    for i in parsed_features.keys():
        feat = tf.decode_raw(parsed_features[i],dtype[i])
        if i == "positions" or i == "mmffpositions":
            feat = tf.reshape(feat,[-1,3])
        if i == "targets":
            feat = tf.reshape(feat,(15,1))
        features_new[i] = feat
    return features_new

def read_tfrecord(tfrecord, index):
    '''

    tfrecord: input name, can be str or list
    n_instance: number of instances will be readed
    random_seed: shuffle random seed
    shuffle: whether shuffle

    '''
    record = tf.data.TFRecordDataset(tfrecord)
    record = record.map(__parse_function)
    record = record.padded_batch(batch_size= 1000,padded_shapes=({"elements":[None,],"targets":[15,1],"positions":[None,3],"mmffpositions":[None,3]}),padding_values=None)
    iterator  = record.make_one_shot_iterator()
    features_new = iterator.get_next()
    with tf.Session() as sess:
   
        it = int(index/1000) + 1
        idx_final = index%1000
   
        for idx, i in enumerate(range(it)):
            if idx < int(it -1):
                sess.run(features_new)
            else:
                feature_int= sess.run(features_new)
                length = len([i for i in feature_int["elements"][idx_final-1] if i != 0])
                feature = {"elements":feature_int["elements"][idx_final -1][0:length],
                           "targets":feature_int["targets"][idx_final -1],
                           "positions":feature_int["positions"][idx_final -1][0:length],
                           "mmffpositions":feature_int["mmffpositions"][idx_final -1][0:length]}
    return feature


def write_tfrecord_fromtfrecord(tfrecord,outfile,data_list):
    element_conversions = {"H":1,"C":6,"O":8,"N":7,"F":9}
    olddir = os.getcwd()
    writer = tf.python_io.TFRecordWriter(outfile)
    features = []
    for idx,i in enumerate(data_list):
        features.append(read_tfrecord(tfrecord, i))
        if idx % 1000 == 0:
            print(str(idx) + "/" + str(len(data_list)))
    assert len(data_list) == len(features)
    for f in features:
        newfeatures = {key: _bytes_feature([f[key].tostring()]) for key in f.keys()}
        example = tf.train.Example(features=tf.train.Features(feature=newfeatures))
        writer.write(example.SerializeToString())
    writer.close()
    return None

    
    



if __name__ == "__main__":
    ### total number of data is 133885
    ### train: 99000 validation:1000 test_live:1000 test:32885
    partition = {"train":99000,"validation":1000,"test_live":1000}
    total_num = 133885
    datadir = "../../data/processed/"
    os.chdir(datadir)
    if not os.path.isfile(os.path.join(datadir,"split_infor.npz")):
        train_id,val_id,test_live_id,test_id = split_data(partition, total_num, False)
    else:
        split = np.load(os.path.join(datadir,"split_infor.npz"))
        train_id = split["train"]
        val_id = split["validation"]
        test_live_id = split["test_live"]
        test_id = split["test"]
    #tar = tarfile.open("../../data/raw/qm9_mmff.tar.bz2")
    tfrecord = "../../data/raw/qm9_mmff.tfrecord"
    outfile = "train.tfrecord"
    write_tfrecord_fromtfrecord(tfrecord,outfile,train_id)
    tar.close()
