#coding=utf-8
from __future__ import division
#本文件用于计算hash列表下的排序结果，使用方法将是semantic hash rank（SHR）和SHR-Cluster（SHR-C）
#这里需要维护一张数据顺序与数据ID的关系表，name_list与dic中序列池的对应
import numpy as np
import math
import pandas as pd
import xlwt
from . import cal_SHR

#汉明距离总长
hl = 48
#限定距离长度
rhl = 24
#限定方式
oper = '>='
#距离计算方式
dis_method = 'Hamming16'
#是否使用限定条件
isRestrict = True
#精度控制
prec = math.pow(10,-8)
#阻尼系数
damping = 1
#打开数据资源文件并读取所有内容
name_list = np.loadtxt(fname="filename.txt",dtype="string")
hash_list = np.loadtxt(fname="filehash.txt",dtype="string")
if len(name_list)!=len(hash_list):
    print("原始数据不匹配")
else:
    #根据hash_list，利用SHR算法计算得到R
    [u_R,u_A,u_dis_m,u_iter_num] = cal_SHR.cal_node_Rank(hash_list,hl,rhl,oper=oper,dis_method=dis_method,isRestrict=isRestrict,damping=damping,prec=prec)

    #根据hash_list，利用SHR-RC算法计算得到R
    [rc_R,rc_A,rc_dis_m,rc_dic,rc_order,rc_iter_num] = cal_SHR.cal_cluster_Rank(hash_list,hl,rhl,oper=oper,dis_method=dis_method,isRestrict=isRestrict,damping=damping,prec=prec)

    #根据增量，利用SHR-RC算法计算得到R,增加一个节点
    #[rc_R,rc_A,rc_dis_m,rc_dic,rc_order,rc_iter_num,name_list] = cal_SHR.cal_cluster_Rank_add_node(name_list,"cctv.js","00000CFFFFFF",rc_A,rc_dis_m,rc_dic,rc_order,hl,rhl,oper=oper,dis_method=dis_method,isRestrict=isRestrict,damping=damping,prec=prec)

    #根据增量，利用SHR-RC算法计算得到R,删除一个节点
    #[rc_R,rc_A,rc_dis_m,rc_dic,rc_order,rc_iter_num,name_list] = cal_SHR.cal_cluster_Rank_del_node(name_list,1,A,dis_m,dic,order,hl,damping=damping,prec=prec)

    #根据增量，利用SHR-RC算法计算得到R,编辑一个节点
    #[rc_R,rc_A,rc_dis_m,rc_dic,rc_order,rc_iter_num,name_list] = cal_SHR.cal_cluster_Rank_update_node(name_list,2,"00000CFCFFFF",rc_A,rc_dis_m,rc_dic,rc_order,hl,rhl,oper=oper,dis_method=dis_method,isRestrict=isRestrict,damping=damping,prec=prec)

    print(u_R)

    print(rc_R)
    #print name_list
    #print rc_order
    pass
    #ResultOUT = pd.DataFrame(rc_R)
    #ResultPATH = r"nodeR.xls"
    #ResultOUT.to_excel(ResultPATH)

    #ResultOUT = pd.DataFrame(u_R)
    #ResultPATH = r"nodeR2.xls"
    #ResultOUT.to_excel(ResultPATH)