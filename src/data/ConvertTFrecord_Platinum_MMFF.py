import tarfile
import tempfile
from tempfile import TemporaryDirectory
import tensorflow as tf
import os,sys
import numpy as np
from ase.units import Hartree, eV, Bohr, Ang, mol, kcal


def readSDF(content):
    elementdict = {'H':1, 'C':6, 'N':7,  'O':8,'F':9}
    content = content.split('\n')
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
    content = content.strip().split('\n')[1:]
    
    return [line.split(',') for line in content]

   

def convertUnits(log):
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

def readTars(tar, idx, tempdir,logcsv):
    
    log = [log for log in logcsv if log[0] == idx][0]

    sdf1 = tar.extract('./' + idx + '.sdf', path = tempdir)
    sdf2 = tar.extract('./' + idx + '.opt.sdf',path = tempdir)

    if log[1] == '':

        return [[],[],[],[]]
    
    else:
        tar.extract('./' + idx + '.sdf', path = tempdir)
        tar.extract('./' + idx + '.opt.sdf',path = tempdir)
        sdf1 = os.path.join(tempdir,'./' + idx + '.sdf')
        sdf2 = os.path.join(tempdir,'./' + idx + '.opt.sdf')
        e1, p1 = readSDF(open(sdf1).read())
        e2, p2 = readSDF(open(sdf2).read())
        log = convertUnits(log)
        logfloat = [float(i) for i in log[1:-1]]

        if len(e1) != len(e2):
            return [[],[],[],[]]
        else:
            return  [e1, p1, p2, logfloat]
            


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def processQMSDF(intar,inpro,outtf,index_list):
    
    writer = tf.python_io.TFRecordWriter(outtf)
    print("The number of confs:" + str(len(index_list)))

    tar = intar
    logfn = inpro
    logcsv = readCSV(open(logfn).read())

    for idx in index_list:
        with TemporaryDirectory() as temp_dir:
            data_infor = readTars(tar, idx, temp_dir, logcsv)
            elements = np.array(data_infor[0]).astype(np.int64)
            positions1 = np.array(data_infor[1]).astype(np.float32).ravel()
            positions2 = np.array(data_infor[2]).astype(np.float32).ravel()
            targets = np.array(data_infor[3]).astype(np.float32)
            if elements != []:
                newfeatures = {'elements' : _bytes_feature(tf.compat.as_bytes(elements.tostring())), 
                               'positions1' : _bytes_feature(tf.compat.as_bytes(positions1.tostring())),
                               'positions2' : _bytes_feature(tf.compat.as_bytes(positions2.tostring())),
                               'targets' : _bytes_feature(tf.compat.as_bytes(targets.tostring()))} 
            
                example = tf.train.Example(features=tf.train.Features(feature=newfeatures))
                writer.write(example.SerializeToString())  
            else:
                print("Error " + idx)
    writer.close()

if __name__ == "__main__":
    outtf = '../../data/processed/Platinum_MMFF.tfrecord'
    intar = tarfile.open("../../data/raw/Platinum_MMFF.tar.bz2")
    inrmsd = "../../data/raw/RMSD.csv"
    inpro = "../../data/raw/Gaussian_properties_allRMSD.csv"
    index_list = [line.split(",")[1].rstrip() for line in open(inrmsd) if line.split(",")[1].rstrip() != "index"]
    processQMSDF(intar,inpro, outtf,index_list)

