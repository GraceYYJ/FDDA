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

        predicthashcode = []
        predicthashstr = []
        starttime = datetime.datetime.now()
        inputs = graph.get_tensor_by_name('images:0')
        hashtags=graph.get_tensor_by_name('hashbit:0')

        predictions = graph.get_tensor_by_name('Accuracy/predictions:0')
        ids, labels, images = Utils.getidsAndimages(__NAME__DATA)
        hashtags = Utils.getHashtags(__NAME__DATA)
        batch_idxs = len(ids) // batchsize

        for idx in range(0, batch_idxs):
            batch = images[idx * batchsize:(idx + 1) * batchsize]
            batch_images = np.array(batch).astype(np.float32)
            hashcode = sess.run(predictions, feed_dict={inputs: batch_images})
            print(hashcode)  # [[ 0.]] Yay!
            predicthashcode.extend(hashcode)
        predicthashcode = np.asarray(predicthashcode, dtype=np.int32)
        print('predicthashcode:',predicthashcode.shape)
        predicthashs = h5py.File(os.path.join(__PATH__DATA, 'predicthasharray.hy'), 'w')
        predicthashs.create_dataset("predicthasharray", data=predicthashcode)
        # predicthashs.create_dataset("originlabel", data=labels)
        print(len(predicthashcode))

        for i in range(len(predicthashcode)):
            strx = "".join(str(j) for j in predicthashcode[i])
            print(predicthashcode[i])
            print(strx)
            predicthashstr.append(strx.encode())
        print(predicthashstr)
        predicthashstr = np.asarray(predicthashstr)
        predicthashstrf = h5py.File(os.path.join(__PATH__DATA, 'predicthashstr.hy'), 'w')
        predicthashstrf.create_dataset("predicthashstr", data=predicthashstr)
        #predicthashstrf.create_dataset("originlabel", data=labels)
    print("finish")
    endtime = datetime.datetime.now()
    usetime = (endtime - starttime).seconds
    print(usetime, "seconds")

    predicthashs.close()
    predicthashstrf.close()
