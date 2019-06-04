#coding=utf-8
from __future__ import division
#本文件用于计算hash列表下的排序结果，使用方法将是semantic hash rank（SHR）和狭义团簇条件下的SHR-RC

import numpy as np
import math
from . import cal_A
from . import cal_Rank
from . import cal_dis_mat
from . import cal_cluster
import time

#全节点rank入口
def rank_node(A,hash_list,damping,prec):
    if damping != 1:
        damping_A = cal_A.cal_node_A_damping(A,damping)
        #print("加了阻尼系数的A：\n",damping_A)
        [R,iter_num] = cal_Rank.iterR(np.mat(damping_A),np.mat(np.ones(len(hash_list))).T,prec)
        print("迭代次数:",iter_num)
    else:
        [R,iter_num] = cal_Rank.iterR(np.mat(A),np.mat(np.ones(len(hash_list))).T,prec)
    return [R,iter_num]

#以全节点方式计算rank
def cal_node_Rank(hash_list,hl,rhl,oper='>=',dis_method='Hamming2',isRestrict=True,damping=1,prec=math.pow(10,-8)):
    #计算得到列表节点的距离矩阵
    nodes_num=len(hash_list)
    dis_m = cal_dis_mat.cal_dis_mat(hash_list,dis_method,hl,rhl,oper,isRestrict=isRestrict)
    #计算得到A矩阵
    astarttime=time.time()
    A = cal_A.cal_node_A(hash_list,dis_m,hl)
    aendtime = time.time()
    print("获取迭代矩阵的时间:", aendtime - astarttime)
    #print("未加阻尼的A：\n",A)
    #测试用输出6次不同的A
    #cal_Rank.num_iterR(np.mat(A),np.mat(np.ones(A.shape[0])).T,6)
    rankstarttime=time.time()
    [R,iter_num] = rank_node(A,hash_list,damping,prec)
    rankendtime=time.time()
    print("算法迭代时间:", rankendtime - rankstarttime)
    print("算法总体时间:",rankendtime - astarttime)
    return [R,A,dis_m,iter_num]

#狭义团簇rank入口
def rank_cluster(A,count,order,sum,damping,prec):
    if damping != 1:
        damping_A = cal_A.cal_cluster_A_damping(A,count,sum,damping)
        [R,iter_num] = cal_Rank.iterR_Cluster(np.mat(damping_A),np.mat(np.ones(len(order))).T,prec,count)
    else:
        [R,iter_num] = cal_Rank.iterR_Cluster(np.mat(A),np.mat(np.ones(len(order))).T,prec,count)
    #在保持维度不变的情况下，将R值大小还原为单节点下情况的值
    R = cal_Rank.restore_node_R(R,sum,len(order))
    return [R,iter_num]

#以狭义团簇方式计算rank
def cal_cluster_Rank(hash_list,hl,rhl,oper='>=',dis_method='Hamming2',isRestrict=True,damping=1,prec=math.pow(10,-8)):
    #计算得到团簇字典和计算列表顺序
    # order是以第一次出现为依据的原始数据顺序 e.g. [1 2 4 3 2 1] => [1 2 4 3] 后面的顺序都遵从order
    [dic,order] = cal_cluster.cal_cluster_list(hash_list)
    sum = cal_cluster.sum_node(dic,order)
    #计算得到团簇节点的距离矩阵
    dis_m = cal_dis_mat.cal_dis_mat(order,dis_method,rhl,oper,isRestrict=isRestrict)
    #计算得到团簇节点下的A矩阵
    [A,count] = cal_A.cal_cluster_A(dic,order,dis_m,hl)
    #cal_Rank.num_cluster_iterR(np.mat(A),np.mat(np.ones(len(order))).T,6,count,len(hash_list),len(order))
    #阻尼系数加工A
    [R,iter_num] = rank_cluster(A,count,order,sum,damping,prec)
    return [R,A,dis_m,dic,order,iter_num]

#在狭义团簇方式下，以增加节点的方式计算rank
def cal_cluster_Rank_add_node(name_list,nodename,node,A,dis_m,dic,order,hl,th,oper='>=',dis_method='Hamming2',isRestrict=True,damping=1,prec=math.pow(10,-8)):
    [A,dic,order,dis_m] = cal_A.add_node(dic,order,dis_m,hl,node,A,th,oper,dis_method,isRestrict)
    name_list = np.append(name_list,nodename)
    count = cal_cluster.count_cluster(dic,order)
    sum = cal_cluster.sum_node(dic,order)
    #阻尼系数加工A
    [R,iter_num] = rank_cluster(A,count,order,sum,damping,prec)
    return [R,A,dis_m,dic,order,iter_num,name_list]

#在狭义团簇方式下，以减少节点的方式计算rank
def cal_cluster_Rank_del_node(name_list,num_i,A,dis_m,dic,order,hl,damping=1,prec=math.pow(10,-8)):
    [A,dic,order,dis_m] = cal_A.del_node(dic,order,dis_m,num_i,hl,A)
    name_list = np.delete(name_list, num_i, axis=0)
    count = cal_cluster.count_cluster(dic,order)
    sum = cal_cluster.sum_node(dic,order)
    #阻尼系数加工A
    [R,iter_num] = rank_cluster(A,count,order,sum,damping,prec)
    return [R,A,dis_m,dic,order,iter_num,name_list]

#在狭义团簇方式下，以修改节点的方式计算rank
def cal_cluster_Rank_update_node(name_list,num_i,node,A,dis_m,dic,order,hl,th,oper='>=',dis_method='Hamming2',isRestrict=True,damping=1,prec=math.pow(10,-8)):
    [A,dic,order,dis_m] = cal_A.del_node(dic,order,dis_m,num_i,hl,A)
    nodename = name_list[num_i]
    name_list = np.delete(name_list, num_i, axis=0)
    [A,dic,order,dis_m] = cal_A.add_node(dic,order,dis_m,hl,node,A,th,oper,dis_method,isRestrict)
    name_list = np.append(name_list,nodename)
    count = cal_cluster.count_cluster(dic,order)
    sum = cal_cluster.sum_node(dic,order)
    #阻尼系数加工A
    [R,iter_num] = rank_cluster(A,count,order,sum,damping,prec)
    return [R,A,dis_m,dic,order,iter_num,name_list]
