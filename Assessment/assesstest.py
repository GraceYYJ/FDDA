# coding=utf-8

import datetime
import time

import tensorflow as tf
import numpy as np
import h5py
import os.path
import argparse

__PATH__TEST = '../datasets/test_images'

__PATH__MODEL = '../DSTH/dsthmodel/checkpoint/cifar10_50_32_32/'
__PATH__DATA = '../DSTH/datasets/cifar10/'
__PATH__RANK = '../SHRANK/rankresult/'

parser = argparse.ArgumentParser(description='assessment parser')
parser.add_argument('-n', type=int)
parser.add_argument('-r', type=int)
parser.add_argument('-w', nargs='+', type=float)


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
    # print("images", images.shape)
    return ids, data, images


def onlineAssess(V, qNum, qImages, qWeights, r, SHRScore):
    weightedScoreSum = 0
    N = len(V)
    hashstr = []
    with tf.Session() as sess:
        # resotore 训练好的模型
        saver = tf.train.import_meta_graph(__PATH__MODEL + 'DSTH.model-59502.meta')
        saver.restore(sess, tf.train.latest_checkpoint(__PATH__MODEL))
        graph = tf.get_default_graph()
        # 根据之前建立模型给每个图节点定义的名字，用graph把这个节点拿出来
        inputs = graph.get_tensor_by_name('images:0')
        predictions = graph.get_tensor_by_name('Accuracy/predictions:0')
        # 生成查询图像的Hash码
        qHashcodes = sess.run(predictions, feed_dict={inputs: np.array(qImages).astype(np.float32)})
        for i in range(len(qHashcodes)):
            strx = "".join(str(j) for j in qHashcodes[i])
            hashstr.append(strx.encode())
            # print(qHashcodes)
    # 根据查询图像的Hash码，在图谱中进行匹配与加权计算
    matchcount = 0
    for i in range(0, qNum):
        matchIndexs = []
        sum = 0
        count = 0
        for j in range(0, N):
            # print(hashstr[i],V[j],hammingDist(hashstr[i], V[j]))
            if hammingDist(hashstr[i], V[j]) <= r:
                matchIndexs.append(j)
                sum = sum + SHRScore[j]
                count = count + 1
                matchcount = matchcount + 1
    if count == 0:
        print("no match data!")
        return
    else:
        print(str(matchcount) + " pic match")
        averageScore = sum / count
        weightedScoreSum = weightedScoreSum + float(qWeights[i]) * averageScore
        Sq = weightedScoreSum / qNum
    sortSHRScore = sorted(SHRScore)
    # 计算Tq
    for k in range(len(sortSHRScore)):
        if sortSHRScore[k] <= Sq:
            Rq = k
        else:
            Rq = N
        Tq = 1 - Rq / N
    return Sq, Tq


def hammingDist(hashstr1, hashstr2):
    """Calculate the Hamming distance between two bit strings"""
    # print(hashstr1,hashstr2)
    assert len(hashstr1) == len(hashstr2)
    return sum(c1 != c2 for c1, c2 in zip(hashstr1, hashstr2))


if __name__ == "__main__":
    ids, labels, images = getidsAndimages(__PATH__DATA)
    # print(images.shape)
    args = parser.parse_args()
    qnum = args.n
    qrange = args.r
    qweights = args.w
    qImages = images[0:qnum]
    GraphV = h5py.File(os.path.join(__PATH__DATA, 'hashstr60w.hy'), 'r')["hashstr60w"].value
    # shr_score= h5py.File(os.path.join(__PATH__RANK, 'cifarSHR.hy'), 'r')["result"]
    shr_score = np.ones([GraphV.shape[0]])
    starttime = time.time()
    # 进行在线评估，传入GraphV：图谱顶点hash码，qImages：查询图像组，
    # weights：查询权重组，qrange：匹配范围，shr_score：图谱SHR分数
    onlineAssess(GraphV, qnum, qImages, qweights, qrange, shr_score)
    endtime = time.time()
    print("assess time", endtime - starttime)
