# coding: utf-8
from __future__ import division
from numpy import *
import numpy as np
import h5py
import datetime
import os.path
import time

__PATH__GRAPH = '../Graph/graphfile/'
__PATH__RANK = './rankresult'

L = 48

acc1 = 0.0000001
acc2 = 7

NUM = 20
num = "20"


class Hotpots:
    def __init__(self, graph_path, rank_path, maxdis):
        self.__PATH__GRAPH = graph_path
        self.__PATH__RANK = rank_path
        self.maxdis=maxdis

    def getLminusdij(self, objectNum):
        Lminusdij = mat(zeros((1, objectNum)))
        sum = 0
        relationfile = h5py.File(os.path.join(self.__PATH__GRAPH, 'hammingrelation' + num + '_' + str(self.maxdis) + '.hy'), 'r')
        relations = relationfile["hammingrelation" + num + "_" + str(self.maxdis)].value
        for i in range(0, objectNum):
            for j in range(0, objectNum):
                if i != j and relations[i][j] != -1:
                    sum = sum + L - relations[i][j]
            Lminusdij[0, i] = sum
            sum = 0
        relationfile.close()
        print(Lminusdij)
        return Lminusdij

    def getFuncNij(self, objectNum):
        Lminusdij = self.getLminusdij(objectNum)
        funcNij = mat(zeros((objectNum, objectNum)))
        relationfile = h5py.File(os.path.join(self.__PATH__GRAPH, 'hammingrelation' + num + '_' + str(self.maxdis) + '.hy'), 'r')
        relations = relationfile["hammingrelation" + num + "_" + str(self.maxdis)].value
        print(relations)

        for i in range(0, objectNum):
            for j in range(0, objectNum):
                if i != j and relations[i][j] != -1:
                    funcNij[i, j] = (L - relations[i][j]) / Lminusdij[0, j]
                    print(funcNij[i,j] ,L,relations[i][j],Lminusdij[0, j])
                else:
                    funcNij[i, j] = 0
        print("------------first funcNij----------")
        print(funcNij)

        for i in range(0, objectNum):
            for j in range(0, objectNum):
                funcNij[i, j] = 0.85 * funcNij[i, j] + 0.15 / objectNum
        print("------------last funcNij----------")
        print(funcNij)

        funcNijfile = h5py.File(os.path.join(self.__PATH__RANK, 'funcNij' + num + '.hy'), 'w')
        funcNijfile.create_dataset("funcNij" + num, data=funcNij)
        funcNijfile.close()
        relationfile.close()
        return funcNij

    def hotpotIter(self, objectNum):
        funcNijfile = h5py.File(os.path.join(self.__PATH__RANK, 'funcNij' + num + '.hy'), 'r')
        funcNij = funcNijfile["funcNij" + num].value
        # print funcNij
        starttime = time.time()
        # print("------------firstR----------")
        R = mat(ones((objectNum, 1)))
        # print R
        i = 0
        Result = np.dot(funcNij, R)
        print("------------firstResult----------")
        print(Result)
        while not self.Equals(Result, R):
            R = Result
            Result = np.dot(funcNij, R)
            i = i + 1
            # print("---------------------------")
            # print i
            print("Result")
            print(Result)
            # print("R")
            # print R
        # print("===============final result==============")
        endtime = time.time()
        usetime = endtime - starttime
        print(i, usetime)
        f = os.file(os.path.join(self.__PATH__RANK, 'itertime' + num + str(acc2) + ".txt"), "a+")
        f.write("迭代次数" + str(i) + '\n' + "耗时：" + str(usetime))
        f.close()
        # print("R")
        # print R
        # print("Result")
        # print Result
        Resultfile = h5py.File(os.path.join(self.__PATH__RANK, 'Result' + num + '_' + str(acc2) + '.hy'), 'w')
        Resultfile.create_dataset("Result", data=Result)
        Resultfile.close()
        return Result

    def Equals(self, vector1, verctor2):
        result = vector1 - verctor2
        flag = True
        for i in range(0, result.shape[0]):
            if not (-acc1 < result[i, 0] < acc1):
                flag = False
        return flag
