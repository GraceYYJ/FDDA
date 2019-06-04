# coding: utf-8
import csv
import time
import h5py
import os.path
import numpy as np
import datetime

__PATH__GRAPH = './graphfile'
L = 48

maxdis = 24

num = "20"
NUM = 20


def hammingDist(hashstr1, hashstr2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(hashstr1) == len(hashstr2)
    return sum(c1 != c2 for c1, c2 in zip(hashstr1, hashstr2))


def hammingdist(hasharray1, hasharray2):
    hammingdist = np.sum(hasharray1 != hasharray2)
    return hammingdist


if __name__ == "__main__":
    # relations
    hyfile_node = h5py.File(os.path.join(__PATH__GRAPH, 'hashnode' + num + '.hy'), 'r')
    hyfile_relation = h5py.File(os.path.join(__PATH__GRAPH, 'hammingrelation' + num + '_' + str(maxdis) + '.hy'), 'w')
    hashcode = hyfile_node["hashnode" + num].value
    hashcode = hashcode[0:NUM]
    relations = np.zeros((NUM, NUM))
    count = 0
    starttime = datetime.datetime.now()
    for i in range(len(hashcode)):
        for j in range(len(hashcode)):
            dist = hammingdist(hashcode[i], hashcode[j])
            if (i != j) and (dist < maxdis):
                print(i, j)
                relations[i][j] = dist
                count = count + 1
            else:
                relations[i][j] = -1
    endtime = datetime.datetime.now()
    print('usetime = ', (endtime - starttime).seconds)
    hyfile_relation.create_dataset("hammingrelation" + num + "_" + str(maxdis), data=relations)
    hyfile_node.close()
    hyfile_relation.close()
