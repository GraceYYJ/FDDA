#coding=utf-8

import datetime

import tensorflow as tf
import numpy as np
import h5py
import os.path

num = 100

__PATH__TEST = '../datasets/test_images'

__PATH__MODEL = '../DSTH/dsthmodel/checkpoint/cifar10_50_32_32/'
__PATH__DATA = '../DSTH/datasets/cifar10/'
__PATH__RANK = '../SHRANK/rankresult/'

def getidsAndimages(dataset_name):
    id_txt = os.path.join(dataset_name, 'id.txt')
    try:
        with open(id_txt, 'r') as fp:
            ids = [s.strip() for s in fp.readlines() if s]
    except:
        raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
    file = os.path.join(dataset_name, 'data.hy')
    try:
        data = h5py.File(file, 'r')
    except:
        raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
    images = []
    for i in range(len(ids)):
        image = data[ids[i]]['image'].value / 255.
        images.append(image.astype(np.float32))
    images = np.asarray(images, dtype=np.float32)
    data.close()
    print("images", images.shape)
    return ids, data, images


def onlineAssess(V, qImages, qWeights, r, SHRScore):
    weightedScoreSum = 0
    qNum = len(qImages)
    N = len(V)
    with tf.Session() as sess:
        # resotore 训练好的模型
        saver = tf.train.import_meta_graph(__PATH__MODEL + 'DSTH.model-59502.meta')
        saver.restore(sess, tf.train.latest_checkpoint(__PATH__MODEL))
        graph = tf.get_default_graph()

        predicthashcode = []
        predicthashstr = []
        # 根据之前建立模型给每个图节点定义的名字，用graph把这个节点拿出来
        inputs = graph.get_tensor_by_name('images:0')
        # 这一个predictions就是最后48位feature二值化后的Hash码值
        predictions = graph.get_tensor_by_name('Accuracy/predictions:0')
        ids, labels, images = getidsAndimages(__PATH__DATA)
        #print(images[0:1])
        qHashcodes = sess.run(predictions, feed_dict={inputs: np.array(qImages).astype(np.float32)})
        print(qHashcodes[0])
    for i in range(0, qNum):
        matchIndexs = []
        sum = 0
        count = 0
        for j in range(0, N):
            # print(j)
            if hammingDist(qHashcodes[i], V[j]) <= r:
                matchIndexs.append(j)
                sum = sum + SHRScore[j]
                count = count + 1
        averageScore = sum / count
        weightedScoreSum = weightedScoreSum + qWeights[i] * averageScore
    Sq = weightedScoreSum / qNum
    sortSHRScore = sorted(SHRScore)
    for k in range(len(sortSHRScore)):
        Rq = k
        Tq = 1 - Rq / N
    return Sq, Tq


def hammingDist(hashstr1, hashstr2):
    """Calculate the Hamming distance between two bit strings"""
    # print(hashstr1,hashstr2)
    assert len(hashstr1) == len(hashstr2)
    return sum(c1 != c2 for c1, c2 in zip(hashstr1, hashstr2))


if __name__ == "__main__":
    # hash_list = ['000110110010000001011010001010110111110111000011',
    #              '000101011010001111011010001010110111110111000011',
    #              '000110110010001111000001111101011011111011100001',
    #              '000110110010001111011010001010110111110111000011']
    ids, labels, images = getidsAndimages(__PATH__DATA)
    testNum=1
    qImages=images[0:testNum]
    #weights = [0.1, 0.2, 0.3, 0.4]
    weights = [1]
    V = h5py.File(os.path.join(__PATH__DATA, 'predicthashstr.hy'), 'r')["predicthashstr"]
    shr_score= h5py.File(os.path.join(__PATH__RANK, 'cifarSHR.hy'), 'r')["result"]
    starttime = datetime.datetime.now()
    print(onlineAssess(V, qImages, weights, 48, shr_score))
    endtime = datetime.datetime.now()
    print("assess time",(endtime-starttime).seconds)
