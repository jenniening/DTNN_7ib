import os, sys
import numpy as np
import tensorflow as tf
from ase.db import connect
from ase.units import kcal, mol
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from ase.units import Hartree, eV, Bohr, Ang, mol, kcal

import tempfile

from rdkit.Chem import AllChem
from rdkit import Chem

from rdkit.Chem import AllChem
from rdkit import Chem
import os, sys
import numpy as np

import logging

import click

elementdict = {1:'H', 6:'C', 7:'N',  8:'O', 9:'F'}
molconvert = "/Applications/MarvinSuite/bin/molconvert"
obabel = ""
### This file conversion without unit conversion or atom reference deduction. 
### Thus, the unit of energy term is still eV and atom_reference still need to be included in the training process.

def writexyz(element, position, filename, props):
    f = open(filename, 'w')
    f.write(str(len(element)) + '\n')
    props = [str(i) for i in props]
    f.write(" ".join(props) + '\n')
    for e, p in zip(element, position):
        line = '{}       {:8.4f} {:8.4f} {:8.4f}\n'.format(elementdict[e], p[0], p[1], p[2])
        f.write(line)
    f.close()


def getEnergyDiff(atom_ref, elements, es):
    eouts = []
    for idx, e in enumerate(es):
        e0 = np.sum(atom_ref[:, idx+1:(idx+2)][elements], 0)
        eouts.append((e-e0)[0])
    return eouts


def convertXYZ2SDF(indb, outdir):

    prop_names = ['rcA', 'rcB', 'rcC', 'mu', 'alpha', 'homo', 'lumo',
                  'gap', 'r2', 'zpve', 'energy_U0', 'energy_U', 'enthalpy_H',
                  'free_G', 'Cv']

    conversions = [1., 1., 1., 1.,  1.,
                   1., 1., 1.,
                   1., 1.,
                   1., 1., 1.,
                   1., 1.]

    count = 0

    with connect(indb) as conn:
        n_structures = conn.count()

        for row in conn.select():
            at = row.toatoms()
            count += 1

            tmpxyz = outdir + str(count) + '.xyz'
            tmpsdf = outdir + str(count) + '.sdf'
            tmpoptsdf = outdir + str(count) + '.mmff.sdf'

            props = [row[pn] * pu for pn, pu in zip(prop_names, conversions)]
            ### No deduction of reference ###
            #tmpprop = props[10:14]
            #tmpprop = getEnergyDiff(atom_ref, at.numbers, tmpprop)
            #props = props[:10] + tmpprop + props[14:]
            #######
            writexyz(at.numbers, at.positions, tmpxyz, props)

            try:

                cmd = obabel + ' -ixyz ' + tmpxyz + ' -osdf -O ' + tmpsdf
                os.system(cmd)
            except:
                ### Structures can't be converted into sdf using obabel ###
                logging.info(str(count) + ": obabel problem")
                cmd = molconvert + ' sdf ' + tmpxyz + '{xyz:} -o ' + tmpsdf
                os.system(cmd)
                

            # load in sdf
            m = Chem.SDMolSupplier(tmpsdf, removeHs=False)[0]
            AllChem.MMFFOptimizeMolecule(m)

            molref = Chem.SDMolSupplier(tmpsdf, removeHs=False)[0]

            heavyatomidx = []
            for a in m.GetAtoms():
                if a.GetAtomicNum() != 1:
                    heavyatomidx.append(a.GetIdx())

            rmsd = Chem.rdMolAlign.AlignMol(m, molref, atomMap = [(k, k) for k in heavyatomidx])
            m.SetProp('RMSD', str(rmsd))
            w = Chem.SDWriter(tmpoptsdf)
            w.write(m)

            if count % 1000 == 0:
                logging.info(str(count) + ' / ' + str(n_structures))
    return

if __name__ == "__main__":
    
    print("test")

    ### Generate MMFF optimized geomerties ###
    dbpath = "../data/process/split_4"
    indb = os.path.join(dbpath, 'train.db')
    outfir = dbpath + "/train/"
    convertXYZ2SDF(indb, outdir)
