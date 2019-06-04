#coding=utf-8
#计算迭代矩阵A的函数
import numpy as np
from . import common

#cal_cluster_list统计数据l中的唯一元素，以及它们的个数和位置
#重组数据为{010110:[count,[index1,index2,...]],...}
def cal_cluster_list(l):
    count = 0
    dic = {}
    order = []
    for x,y in enumerate(l):
        if not dic.get(y):
            dic[y] = [1,[x]]
            order.append(y)
        else:
            dic[y][0] = dic[y][0]+1
            dic[y][-1].append(x)
    return [dic,order]

#统计order顺序下每个hash所代表的元素个数
def count_cluster(dic,order):
    count = []
    for index,value in enumerate(order):
        count.append(dic[value][0])
    return count

#统计实际元素个数
def sum_node(dic,order):
    return np.sum(count_cluster(dic,order))

#找到recdigit中所有索引号大于num_i的数-1,前移1个数
def dic_index_forward_move(dic,num_i):
    for key in dic:
        for i,v in enumerate(dic[key][1]):
            if v > num_i:
                dic[key][1][i] = v-1
    return dic

#找到recdigit中所有索引号大于等于num_i的数+1,后移1个数
def dic_index_backward_move(dic,num_i):
    for key in dic:
        for i,v in enumerate(dic[key][1]):
            if v >= num_i:
                dic[key][1][i] = v+1
    return dic

#判断某节点是否是团簇
def check_node_isCluster(dic,node):
    return [True,dic[node][0]] if dic[node][0] > 1 else [False,1]