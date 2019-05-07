import tarfile
import tensorflow as tf
import os,sys
import numpy as np
from ase.units import Hartree, eV, Bohr, Ang, mol, kcal
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(total_file, test_size, val_size, test_live_size, random_state = 1, save = True):
    '''Split conformation data based on molecule ID
    
    :param total_file: total_index_file
    :type total_file: csv file
    :param test_size: number of molecules in test set
    :type test_size: int
    :param val_size: number of molecules in validation set
    :type val_size: int
    :param test_live_size: number of molecules in test_live set
    :type test_live_size: int
    :param random_state: defaults to 1
    :param random_state: int, optional
    :param save: defaults to True
    :param save: bool, optional
    :return: training, validation, test_live, test datasets
    :rtype: dataframe
    '''

    total_data = pd.read_csv(total_file)
    print("Total total number of confs:" + str(total_data.shape[0]))
    ### if molecule only has one conformation, this molecule will be put in training set ###
    train_id_1 = total_data[total_data["size"] == 1]["molecule_id"]

    other_data = total_data[~total_data["molecule_id"].isin(train_id_1)]
    ### split dataset into training and testing using molecule id ###
    other_molecules = other_data.groupby("molecule_id").first()
    other_molecules.reset_index(inplace = True)
    train_idx, test_idx = train_test_split(other_molecules, test_size = test_size, random_state=random_state)
    ### split training into training and validation ###
    train_idx, val_idx = train_test_split(train_idx, test_size = val_size, random_state=1)
    ### split testing into testing and test_live ###
    test_idx, test_live_idx = train_test_split(test_idx, test_size = test_live_size, random_state=1)
    ### get final index for conformation ###
    traindf = total_data.loc[(total_data['molecule_id'].isin(train_idx["molecule_id"]))|(total_data['molecule_id'].isin(train_id_1))][["molecule_id","index"]]
    valdf = total_data.loc[total_data['molecule_id'].isin(val_idx["molecule_id"])][["molecule_id","index"]]
    testdf = total_data.loc[total_data['molecule_id'].isin(test_idx["molecule_id"])][["molecule_id","index"]]
    testlivedf = total_data.loc[total_data['molecule_id'].isin(test_live_idx["molecule_id"])][["molecule_id","index"]]

    if save:
        np.savez("split_infor.npz",
                train_mol = traindf["molecule_id"].values, train_idx = traindf["index"].values,
                test_mol = testdf["molecule_id"].values, test_idx = testdf["index"].values,
                validation_mol = valdf["molecule_id"].values, validation_idx = valdf["index"].values,
                test_live_mol = testlivedf["molecule_id"].values, test_live_idx = testlivedf["index"].values)

    return traindf, valdf, testdf, testlivedf


def readSDF(content):
    '''Read molecule information from tarfile 
    
    :param content: sdf file (needed to be decoded)
    :type content: file
    :return: elements, positions
    :rtype: list
    '''

    elementdict = {'H':1, 'C':6, 'N':7,  'O':8,'F':9}
    content = str(content, 'utf-8').split('\n')
    natom = int(content[3].split()[0])
    atoms = content[4:natom+4]
    
    elements = []
    positions = []
    for atom in atoms:
        atom = atom.split()
        elements.append(elementdict[atom[3]])
        positions.append([float(pos) for pos in atom[:3]])
        
    return elements, positions

def readCSV(content):
    '''Read properties from tarfile
    
    :param content: tarfile
    :type content: file
    :return: properties
    :rtype: list
    '''

    content = str(content, 'utf-8').strip().split('\n')[1:]
    
    return [line.split(',') for line in content]
   

def convertUnits(log):
    '''Convert units for properties (Only energy unites have been converted)
    
    :param log: properties (0 is index, others are properties)
    :type log: list
    :return: properties after conversion
    :rtype: list
    '''

    conversions = [1., 1., 1., 1., 1., 1.,
                   Hartree/eV, Hartree/eV, Hartree/eV,
                   1., Hartree/eV,
                   Hartree/eV, Hartree/eV, Hartree/eV,
                   Hartree/eV, 1., Hartree/eV]
    conversions_dict = {}
    for i in range(17):
        conversions_dict[i] = conversions[i]
    log_new = [log[0]]
    log_new.extend([float(log[i])*conversions_dict[i] for i in range(1,17)])

    return log_new

def readTars(datafd, idx, index_list):
    ''' Read files from tarfile
    
    :param datafd: tarfile for whole dataset
    :type datafd: tarfile
    :param idx: inner tarfile index (go through from 1 to 249)
    :type idx: int
    :param index_list: index list for train, validation, test_live and test
    :type index_list: list
    '''

    start = (idx-1) * 1000 + 1
    end = idx * 1000
    logfn = 'sdf' + str(idx) + '/' + str(start) + '_' + str(end) + '.log.csv'
    rmsdfn = 'sdf' + str(idx) + '/' + str(start) + '_' + str(end) + '.rmsd.csv'
    
    tarfn = datafd + 'sdf' + str(idx) + '.tar'
    tar = tarfile.open(tarfn)
    logcsv = readCSV(tar.extractfile(logfn).read())

    for log in logcsv:
        #print(log[0])
    ### only contain structures in index_list ###
        if int(log[0]) in index_list:
            print(log[0])
            sdf1 = 'sdf' + str(idx) + '/' + log[0] + '.sdf'
            sdf2 = 'sdf' + str(idx) + '/' + log[0] + '.opt.star.sdf'
            if log[1] == '':
                continue
            e1, p1 = readSDF(tar.extractfile(sdf1).read())
            e2, p2 = readSDF(tar.extractfile(sdf2).read())
            log = convertUnits(log)
            logfloat = [float(i) for i in log[1:-1]]
            if len(e1) != len(e2):
                print("Error:" + sdf1)
            yield (e1, p1, p2, logfloat)
    tar.close()

    return None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def processQMSDF(inf,outtf, index_list):
    
    writer = tf.python_io.TFRecordWriter(outtf)
    print("The number of confs:" + str(len(index_list)))
    ### go through all inner tarfiles ### 
    max_num = int(np.ceil(max(index_list)/1000) + 1)
    min_num = int(np.ceil(min(index_list)/1000))
    for idx in range(min_num, max_num):
        for data in readTars(inf, idx, index_list):
            elements = np.array(data[0]).astype(np.int64)
            positions1 = np.array(data[1]).astype(np.float32).ravel()
            positions2 = np.array(data[2]).astype(np.float32).ravel()
            targets = np.array(data[3]).astype(np.float32)
            
            newfeatures = {'elements' : _bytes_feature(tf.compat.as_bytes(elements.tostring())), 
                        'positions1' : _bytes_feature(tf.compat.as_bytes(positions1.tostring())),
                        'positions2' : _bytes_feature(tf.compat.as_bytes(positions2.tostring())),
                        'targets' : _bytes_feature(tf.compat.as_bytes(targets.tostring()))}
            
            example = tf.train.Example(features=tf.train.Features(feature=newfeatures))
            writer.write(example.SerializeToString())  
    writer.close()

if __name__ == "__main__":  
    total_file = "../../data/raw/eMol9_mmff_index.csv"
    test_size = 1348
    val_size = 500
    test_live_size = 500
    datadir = "../../data/processed"
    os.chdir(datadir)
    if not os.path.isfile(os.path.join(datadir, "eMol9_infor.npz")):
        traindf, valdf, testdf, testlivedf = split_data(total_file, test_size, val_size, test_live_size)
        index_list = traindf["index"].values[0:5]
    else:
        data = np.load("eMol9_infor.npz")
        index_list = data["train_idx"][0:5]

    file_type = "train"
    outtf = 'eMol9_mmff_' + file_type + '.tfrecord'
    
    inf = "../../data/raw/eMol9_MMFF/"
    processQMSDF(inf,outtf,index_list)
    inf.close()
