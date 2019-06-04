#coding=utf-8
from __future__ import division
import numpy as np
import copy
from . import cal_cluster

#判断节点值对于原矩阵的存在
def check_node_hash(dic,order,hash):
    return [order.index(hash), count_node_hash(dic,hash), cal_cluster.sum_node(dic, order)] if hash in order else [-1, 0, cal_cluster.sum_node(dic, order)]

#找到hash对应的个数
def count_node_hash(dic,hash):
    return dic[hash][0]

#找到num_i对应hash的存在状态
def check_node_index(dic,order,num_i):
    for key in dic:
        if num_i in dic[key][1]:
            return [order.index(key),key,count_node_hash(dic,key)]

#cal_elements_dis计算a和b之间的距离，用method进行方法控制
#&, |, ^表示二进制的AND, OR, XOR运算
#除Hamming中送入的是字符串外，其余送入的都是向量，else里是切比雪夫距离
def cal_elements_dis(a,b,method):
    if method == 'Hamming2':
        return bin(int(a,2)^int(b,2)).count('1')
    if method == 'Hamming16':
        return bin(int(a,16)^int(b,16)).count('1')
    elif method == 'Consine':
        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    elif method == 'Euclidean':
        return np.sqrt(np.sum(np.square(a-b)))
    elif method == 'Manhattan':
        return np.sum(np.abs(a-b))
    else:
        return np.max(np.abs(a-b))

#根据th与oper决定是否将一个数置为-1
def revalue_distance(value,th,oper):
    if oper=='>=':
        return -1 if value>=th else value
    elif oper=='>':
        return -1 if value>th else value
    elif oper=='<=':
        return -1 if value<=th else value
    else:
        return -1 if value<th else value

#根据th判断节点情况，revalue距离矩阵
def revalue_distance2(value,th,oper):
    if oper=='>=':
        return -1 if value>=th else value
    elif oper=='>':
        return -1 if value>th else value
    elif oper=='<=':
        return -1 if value<=th else value
    else:
        return -1 if value<th else value

#对距离矩阵mat进行加工，对于阈值th条件设置下me方式的无意义数据设置-1
def redefine_mat_distance(mat,th,me='>='):
    if me=='>=':
        return map(lambda x:[[i,-1][i>=th] for i in x],mat)
    elif me=='>':
        return map(lambda x:[[i,-1][i>th] for i in x],mat)
    elif me=='<=':
        return map(lambda x:[[i,-1][i<=th] for i in x],mat)
    else:
        return map(lambda x:[[i,-1][i<th] for i in x],mat)

#对mat进行列向归一化处理，当对角系数为0时,此列不需要进行归一化
#为了保留归一化前的矩阵用于后续的研究，这里使用深拷贝进行矩阵运算
def uniform_mat_col(mat):
    cmat = copy.deepcopy(mat)
    colsum=np.sum(cmat,axis=0)
    for j in range(0,len(cmat)):
        if cmat[j][j] != 0:
            for i in range(0,len(cmat)):
                cmat[i][j] = cmat[i][j]/colsum[j]
    return cmat

#单点情况下计算每一列的汉明距离和
def sum_node_mat_hor(en,j,dis_m,hl):
    sum = 0
    for i in range(0,en):
        if dis_m[i][j] != -1:
            if i!=j:
                sum = sum + hl - dis_m[i][j]
    return sum

#团簇条件下按照单点统计汉明距离下列和
def sum_cluster_mat_hor(en,order_count,j,dis_m,hl):
    sum = 0
    for i in range(0,en):
        if dis_m[i][j] == 0:
            sum = sum + hl*(order_count[i]-1)
        elif dis_m[i][j] != -1:
            sum = sum + (hl - dis_m[i][j])*order_count[i]
    return sum

#在原矩阵上通过行向量和列向量进行扩展
def extend_mat_rc(mat,newrow,newcol,cornerValue):
    return np.c_[np.r_[mat,[newrow]],np.r_[newcol,[cornerValue]]]

#按照r行c列去掉一行和一列得到新矩阵,index为[r,c]
def remove_mat_rc(mat,index):
    mat = np.delete(mat,index[0],axis=0)
    mat = np.delete(mat,index[1],axis=1)
    return mat


