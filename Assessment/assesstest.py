import datetime

import tensorflow as tf
import numpy as np
import h5py
import os.path

num = 100

__PATH__TEST = '../datasets/test_images'

__PATH__MODEL = '../DSTH/dsthmodel/checkpoint/cifar10_50_32_32/'
__PATH__DATA = '../DSTH/datasets/cifar10/'
__NAME__DATA = 'cifar10'


def onlineAssess(V, qImages, qWeights, qhashcode, r, SHRScore):
    weightedScoreSum = 0
    qNum = len(qImages)
    N = len(V)
    starttime = datetime.datetime.now()
    # with tf.Session() as sess:
    #     # resotore 训练好的模型
    #     saver = tf.train.import_meta_graph(__PATH__MODEL + 'DSTH.model-59502.meta')
    #     saver.restore(sess, tf.train.latest_checkpoint(__PATH__MODEL))
    #     graph = tf.get_default_graph()
    #
    #     predicthashcode = []
    #     predicthashstr = []
    #     # 根据之前建立模型给每个图节点定义的名字，用graph把这个节点拿出来
    #     inputs = graph.get_tensor_by_name('images:0')
    #     # 这一个predictions就是最后48位feature二值化后的Hash码值
    #     predictions = graph.get_tensor_by_name('Accuracy/predictions:0')
    #     qHashcodes = sess.run(predictions, feed_dict={inputs: np.array(qImages).astype(np.float32)})
    for i in range(0, qNum):
        matchIndexs = []
        sum = 0
        count = 0
        for j in range(0, N):
            # print(j)
            if hammingDist(qhashcode[i], V[j]) <= r:
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
    hash_list = ['000110110010000001011010001010110111110111000011',
                 '000101011010001111011010001010110111110111000011',
                 '000110110010001111000001111101011011111011100001',
                 '000110110010001111011010001010110111110111000011']
    shr_score = [1, 2, 1, 3, 2, 3, 1, 2, 3, 4]
    weights = [0.1, 0.2, 0.3, 0.4]
    hashfile = h5py.File(os.path.join(__PATH__DATA, 'predicthashstr.hy'), 'r')
    V = hashfile["predicthashstr"].value[0:10]
    print(onlineAssess(V, hash_list, weights, hash_list, 48, shr_score))
