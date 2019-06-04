# -*- coding:utf-8 -*-
import argparse
import tensorflow as tf
import datetime
import numpy as np
import h5py
import os.path

import sys

sys.path.append('/tfproject/FDDA')
from DSTH.Common.util import Utils

__PATH__MODEL = '../dsthmodel/checkpoint/cifar10_50_32_32/'
__PATH__DATA = '../datasets/cifar10/'
__NAME__DATA = 'cifar10'

batchsize = 100

if __name__ == '__main__':
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(__PATH__MODEL + 'DSTH.model-59502.meta')
        saver.restore(sess, tf.train.latest_checkpoint(__PATH__MODEL))
        graph = tf.get_default_graph()

        for op in graph.get_operations():
            print(op.name, op.values())

        features = []
        starttime = datetime.datetime.now()
        inputs = graph.get_tensor_by_name('images:0')
        # predictions=graph.get_tensor_by_name('Accuracy/predictions:0')
        # fc4 = graph.get_tensor_by_name('network32/n_fc/add:0')
        slice15 = graph.get_tensor_by_name('network32/n_slice/concat_14:0')
        ids, labels, images = Utils.getidsAndimages(__PATH__DATA)
        batch_idxs = len(ids) // batchsize
        for idx in range(0, batch_idxs):
            batch = images[idx * batchsize:(idx + 1) * batchsize]
            batch_images = np.array(batch).astype(np.float32)
            features48 = sess.run(slice15, feed_dict={inputs: batch_images})
            print(features48)  # [[ 0.]] Yay!
            features.extend(features48)
        features = np.asarray(features, dtype=np.float32)
        print(features)
        print(features.shape)
        predictfeatures48 = h5py.File(os.path.join(__PATH__DATA, 'features48.hy'), 'w')
        predictfeatures48.create_dataset("features48", data=features)
        # predictfeatures48.create_dataset("originlabel", data=labels)
        print("finish")
        endtime = datetime.datetime.now()
        usetime = (endtime - starttime).seconds
        print(usetime, "seconds")

        predictfeatures48.close()
