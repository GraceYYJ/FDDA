import h5py
import os.path
import numpy as np

__PATH__DATA = '../DSTH/datasets/cifar10/'

# GraphV = h5py.File(os.path.join(__PATH__DATA, 'predicthashstr.hy'), 'r')["predicthashstr"].value
# extendGraphV = GraphV
# for i in range(9):
#     extendGraphV = np.concatenate((extendGraphV, GraphV), axis=0)
#
# extendf = h5py.File(os.path.join(__PATH__DATA, 'hashstr60w.hy'), 'w')
# extendf.create_dataset("hashstr60w",data=extendGraphV)
# extendf.close()

print(["===================read hashcode.hy"])
f = h5py.File(os.path.join(__PATH__DATA, 'hashstr60w.hy'), 'r')
print(f["hashstr60w"].name)
print(f["hashstr60w"].value)
print(f["hashstr60w"].value.shape)
f.close()