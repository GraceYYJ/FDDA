# coding=utf-8
from __future__ import division
# 本文件用于计算hash列表下的排序结果，使用方法将是semantic hash rank（SHR）和SHR-Cluster（SHR-C）
# 这里需要维护一张数据顺序与数据ID的关系表，name_list与dic中序列池的对应
import numpy as np
import math
import h5py
import os.path
import numpy as np
#from SHRank.SHR2 import cal_SHR
import cal_SHR
__PATH__DATA = '../../DSTH/datasets/cifar10'
__PATH__RANK = './rankresult'

NUM = 60000

# 汉明距离总长
hl = 48
# 限定距离长度
rhl = 24
# 限定方式
oper = '>='
# 距离计算方式
dis_method = 'Hamming2'
# 是否使用限定条件
isRestrict = True
# 精度控制
prec = math.pow(10, -8)
# 阻尼系数
damping = 0.85

# 打开数据资源文件并读取所有内容
hashfile = h5py.File(os.path.join(__PATH__DATA, 'predicthashstr.hy'), 'r')
hashcode = hashfile["predicthashstr"].value
hashcode = hashcode[0:NUM]
#print(hashcode[0], hashcode.shape)

# 根据hash_list，利用SHR算法计算得到R
# for i in range(10):
#     [u_R, u_A, u_dis_m, u_iter_num] = cal_SHR.cal_node_Rank(hashcode, hl, rhl, oper=oper, dis_method=dis_method,
#                                                             isRestrict=isRestrict, damping=damping, prec=prec)
[u_R, u_A, u_dis_m, u_iter_num] = cal_SHR.cal_node_Rank(hashcode, hl, rhl, oper=oper, dis_method=dis_method,
                                                        isRestrict=isRestrict, damping=damping, prec=prec)
print(u_R[0],len(u_R))
Resultfile = h5py.File(os.path.join(__PATH__RANK, 'cifarSHR.hy'), 'w')
Resultfile.create_dataset("result", data=u_R)
Resultfile.close()
