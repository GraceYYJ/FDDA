#coding=utf-8
from __future__ import division
#迭代计算得到最终的排名分数copy
import copy
import numpy as np
from . import common

def num_iterR(A,R,num):
    while num>0:
        R = A*R
        num = num - 1
        print(R)

def num_cluster_iterR(A,R,num,order_count,org_len,cluster_len):
    while num>0:
        R = A*R
        num = num - 1
        print (R/np.mat(order_count).T)*org_len/cluster_len

#A是迭代矩阵，R是初始化列向量，c是收敛精度
def iterR(A,R,c):
    #tempR = copy.deepcopy(R)
    iter_num = 0
    while True:
        if common.cal_elements_dis(R,A*R,method='Manhattan') > c:
            #print(R)
            R = A*R
            iter_num = iter_num + 1
        else:
            return [R,iter_num]

def iterR_Cluster(A,R,c,order_count):
    [R,iter_num] = iterR(A,R,c)
    return [R/np.mat(order_count).T,iter_num]

def restore_node_R(R,org_len,cluster_len):
    return R*org_len/cluster_len


