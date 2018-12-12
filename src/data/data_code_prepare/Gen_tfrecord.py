import tensorflow as tf
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def predictLayer(inpath, outtf, n = 1000):
    
    # feature dict for model input

    writer = tf.python_io.TFRecordWriter(outtf)
    
    for i in range(n):
        tmpsdf = inpath + str(i+1) + '.sdf'
        tmpmmff = inpath + str(i+1) + '.mmff.sdf'
        
        with open(tmpsdf, 'r') as f:
            targets = f.readlines()[0].split()
        targets = [float(i) for i in targets]
        
        elements = []
        
        m1 = Chem.SDMolSupplier(tmpsdf, removeHs=False)[0]
        m2 = Chem.SDMolSupplier(tmpmmff, removeHs=False)[0]
        
        heavyatomidx = []
        for atom in m1.GetAtoms():
            elements.append(atom.GetAtomicNum())
            if atom.GetAtomicNum() != 1:
                heavyatomidx.append(atom.GetIdx())            
        rmsd = Chem.rdMolAlign.AlignMol(m1, m2, atomMap = [(k, k) for k in heavyatomidx])
        
        
        prop = AllChem.MMFFGetMoleculeProperties(m1)

        #if prop is None:
        #    print(i)
        #    continue
        
        #atomtypes = []
        #for atom in m1.GetAtoms():
        #    atomtypes.append(prop.GetMMFFAtomType(atom.GetIdx()))
        
        
        positions = []
        for atom in m1.GetAtoms():
            pos  = m1.GetConformer().GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
  
        mmffpositions = []
        for atom in m2.GetAtoms():
            pos  = m2.GetConformer().GetAtomPosition(atom.GetIdx())
            mmffpositions.append([pos.x, pos.y, pos.z])
        
        elements = np.array(elements).astype(np.int64)
        #atomtypes = np.array(atomtypes).astype(np.int64)
        targets = np.array(targets).astype(np.float32)
        positions = np.array(positions).astype(np.float32)
        mmffpositions = np.array(mmffpositions).astype(np.float32)
        extras = np.array([rmsd]).astype(np.float32)



        newfeatures = {'elements' : _bytes_feature(tf.compat.as_bytes(elements.tostring())), 
                       #'atomtypes' : _bytes_feature(tf.compat.as_bytes(atomtypes.tostring())), 
                       'positions' : _bytes_feature(tf.compat.as_bytes(positions.ravel().tostring())),
                       'mmffpositions' : _bytes_feature(tf.compat.as_bytes(mmffpositions.ravel().tostring())),
                       'targets' : _bytes_feature(tf.compat.as_bytes(targets.tostring())), 
                       'extras': _bytes_feature(tf.compat.as_bytes(extras.tostring()))}
            
        example = tf.train.Example(features=tf.train.Features(feature=newfeatures))
        writer.write(example.SerializeToString())
            
        if i % 1000 == 0:
            print(str(i) + ' / ' + str(n))

    #sess.close()    
    writer.close()
    
    return 


if __name__ == "__main__":
    inpath = "db/split_2/train/"
    outtf = "db/split_2/train_4000.tfrecord"
    predictLayer(inpath, outtf,n = 4000)
