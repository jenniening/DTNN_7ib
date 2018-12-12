import os
import logging
from random import shuffle

import numpy as np
from ase.db import connect


def split_ase_db(asedb, dstdir, partitions, selection=None):
    partition_ids = list(partitions.keys())
    partitions = np.array(list(partitions.values()))
    if len(partitions[partitions < -1]) > 1:
        raise ValueError(
            'There must not be more than one partition of unknown size!')

    with connect(asedb) as con:
        ids = []
        for row in con.select(selection=selection):
            ids.append(row.id)

    ids = np.random.permutation(ids)
    n_rows = len(ids)
    # to tell the value of partitions in the (0,1) or not : r should be numpy array with true or false
    r = (0. < partitions) * (partitions < 1.)
    # if r is true, the value of partitions[r] will be timed with n_rows
    partitions[r] *= n_rows
    partitions = partitions.astype(np.int)

    if np.any(partitions < 0):
        remaining = n_rows - np.sum(partitions[partitions > 0])
        partitions[partitions < 0] = remaining

    if len(partitions[partitions < 0]) > 1:
        raise ValueError(
            'Size of the partitions has to be <= the number of atom rows!')

    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
    else:
        raise ValueError('Split destination directory already exists:',
                         dstdir)

    split_dict = {}
    with connect(asedb) as con:
        offset = 0
        if partition_ids is None:
            partition_ids = list(range(len(partitions)))
        for pid, p in zip(partition_ids, partitions):
            with connect(os.path.join(dstdir, pid + '.db')) as dstcon:
                print(offset, p)
                split_dict[pid] = ids[offset:offset + p]
                for i in ids[offset:offset + p]:
                    row = con.get(int(i))
                    if hasattr(row, 'data'):
                        data = row.data
                    else:
                        data = None
                    dstcon.write(row.toatoms(),
                                 key_value_pairs=row.key_value_pairs,
                                 data=data)
            offset += p
    np.savez(os.path.join(dstdir, 'split_ids.npz'), **split_dict)

if __name__ == "__main__":
    datadir_raw = "../data/raw"
    datadir_pro = "../data/process"
    asedb = datadir_raw + "/gdb9.db"
    ### split data into training, validation, test_live (which is also been included in final test) and test set ###
    ### repeat 5 times, our final results is the average of these five splits ###
    for i in range(5):
        logging.info("Split Data: split_" + str(i+1) + " start")
        dstdir = datadir_process + "/split_" + str(i+1)
        partitions = {"train":99000,"validation":1000,"test_live":1000,"test":-1}
        split_ase_db(asedb,dstdir,partitions)
        logging.info("split_" + str(i+1) + " finish"))
    logging.info("Done. ")
        

