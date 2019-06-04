# coding: utf-8
import csv
import time
import h5py
import os.path
import numpy as np
import datetime
#from neo4j.v1 import GraphDatabase, basic_auth
__PATH__DATA = '../DSTH/datasets/cifar10'
__PATH__GRAPH = './graphfile'
L=48

num="20"
NUM=20

if __name__ == "__main__":
    hashfile = h5py.File(os.path.join(__PATH__DATA, 'predicthasharray.hy'), 'r')
    hyfile_node = h5py.File(os.path.join(__PATH__GRAPH,'hashnode'+num+'.hy'), 'w')
    hashcode=hashfile["predicthasharray"].value
    hashcode=hashcode[0:NUM]
    #label=hashstrfile["originlabel"].value
    #label=label[0:NUM]
    print(hashcode[0],hashcode.shape)
    hyfile_node.create_dataset("hashnode"+num,data=hashcode)
    #hyfile_node.create_dataset("hashnodearr"+num+"labels",data=label)
    hyfile_node.close()
    hashfile.close()





