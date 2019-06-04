#coding=utf-8
from __future__ import division
#计算迭代矩阵A的函数
import numpy as np
# from . import cal_cluster
# from . import cal_dis_mat
# from . import common
import cal_cluster
import cal_dis_mat
import common

#all_unit_cal将所有点单独计算
def cal_node_A(list,dis_m,hl):
    #节点个数en
    en = len(list)
    A = np.zeros((en,en))
    for j in range(0,en):
        Amother = cal_dis_mat.sum_node_ham_dis_hor(en,j,dis_m,hl)
        for i in range(0,en):
            if dis_m[i][j]==-1 or i==j:
                A[i][j] = 0
            else:
                A[i][j] = (hl-dis_m[i][j])/Amother
    return A

#计算单节点阻尼系数下的A
def cal_node_A_damping(A,damping):
    return damping*A + (1-damping)/A.shape[0]*np.ones((A.shape[0],A.shape[0]))

#combine_same_cal将相同点视为团簇进行计算
def cal_cluster_A(dic,order,dis_m,hl):
    count = cal_cluster.count_cluster(dic,order)
    en = len(order)
    A = np.zeros((en,en))
    #逐列进行分子分母构建
    for j in range(0,en):
        Colmother = cal_dis_mat.sum_cluster_ham_dis_hor(en,count,j,dis_m,hl)
        for i in range(0,en):
            if dis_m[i][j]!=-1:
                if i==j:
                    A[i][j] = (count[i]-1)*hl/Colmother
                else:
                    A[i][j] = count[i]*(hl-dis_m[i][j])/Colmother
    return [A,count]

#计算团簇节点阻尼系数下的A
def cal_cluster_A_damping(A,count,org_num,damping):
    return damping*A + (1-damping)/org_num*(np.mat(count).T*np.ones(len(count)))

#增加一个节点，以增量方式进行计算，这里我们被增量的基础为归一化后的A，增量计算的结果也是归一化的结果
#recdigit用于判断hash是否在其中，dis_m用于更新recdigit，hash为新加入的节点值，A为基础A
#在计算返回A时，要同时更新recdigit和dis_m
def add_node(dic,order,dis_m,hl,hash,A,th,oper,dis_method,isRestrict):
    #node在recdigit中的index，node在orgdigit中有几个，orgdigit中的数据总量
    [index,counter,sumCount] = common.check_node_hash(dic,order,hash)
    en = len(order)
    #当节点存在与原列表中
    if index != -1:
        count = cal_cluster.count_cluster(dic,order)
        for j in range(0,en):
            #判断该列与此节点的连接关系
            #当有连接时
            if dis_m[index][j] != -1:
                caOldmother = cal_dis_mat.sum_cluster_ham_dis_hor(en,count,j,dis_m,hl)
                caNewmother = caOldmother + hl - dis_m[index][j]
                for i in range(0,en):
                    A[i][j] = A[i][j]*caOldmother/caNewmother
                    if i==index:
                        if j!=index:
                            A[i][j] = A[i][j]*(counter+1)/counter
                        else:
                            if counter!=1:
                                A[i][j] = A[i][j]*counter/(counter-1)
                            else:
                                A[i][j] = hl/caNewmother
        # dis_m不变，order不变，recdigit中key为node的count+1并且添加sumCount到位置列表
        dic[order[index]][0] = dic[order[index]][0] + 1
        dic[order[index]][1].append(sumCount)
    else:
        #更新距离矩阵，此时实际维度为en+1
        dis_m = cal_dis_mat.update_dis_mat_add(order,hash,dis_m,th,oper,dis_method,isRestrict)
        order.append(hash)
        dic[hash]=[1,[sumCount]]
        newcol = np.zeros(en)
        newrow = np.zeros(en)
        count = cal_cluster.count_cluster(dic,order)
        newNewmother = cal_dis_mat.sum_cluster_ham_dis_hor(en+1,count,en,dis_m,hl)
        #根据是否连接更新原矩阵，同时以游标更新新行与新列
        for j in range(0,en):
            if dis_m[en][j] != -1:
                caOldmother = cal_dis_mat.sum_cluster_ham_dis_hor(en,count,j,dis_m,hl)
                caNewmother = caOldmother + hl - dis_m[en][j]
                newrow[j] = (hl-dis_m[en][j])/caNewmother
                newcol[j] = common.count_node_hash(dic,order[j])*(hl-dis_m[en][j])/newNewmother
                for i in range(0,en):
                    A[i][j] = A[i][j]*caOldmother/caNewmother
        #将内容更新的A与添加的行与列进行整合
        A = common.extend_mat_rc(A,newrow,newcol,0)

    return [A,dic,order,dis_m]

# 删除一个节点
# 删除节点有3种可能，在删除一个节点后
# 1 这个节点对应的hash还是一个大于两个节点的团簇
# 2 这个节点对应的hash从团簇变成了单个节点
# 3 彻底消失
# np.delete(a,index,axis=0)删除a矩阵的第index行
# np.delete(a,index,axis=1)删除a矩阵的第index列
# 其中，index是数组时，代表多行或者多列
def del_node(dic,order,dis_m,num_i,hl,A):
    #根据num_i,返回对应的hash值以及含有node的个数
    [index,hash,counter] = common.check_node_index(dic,order,num_i)
    count = cal_cluster.count_cluster(dic,order)
    en = len(order)
    if counter >= 2:
        #在>=2的情况下,删除一个节点不需要更新A的维度,order和dis_m
        #recdigit中要找到num_i所在的位置,去掉num_i,并将hash所指代的counter-1
        for j in range(0,en):
            if dis_m[index][j] != -1:
                caOldmother = cal_dis_mat.sum_cluster_ham_dis_hor(en,count,j,dis_m,hl)
                caNewmother = caOldmother - hl + dis_m[index][j]
                for i in range(0,en):
                    A[i][j] = A[i][j]*caOldmother/caNewmother
                    if i == index:
                        if j!=index:
                            A[i][j] = A[i][j]*(counter-1)/counter
                        else:
                            A[i][j] = A[i][j]*(counter-2)/(counter-1)
        dic[hash][0] = counter - 1
        dic[hash][1].remove(num_i)
    else:
        #在=1的情况下，删除一个节点需要更新A的维度,order和dis_m
        #recdigit中要去掉num_i所在位置的字典
        for j in range(0,en):
            if dis_m[index][j] != -1:
                if j != index:
                    caOldmother = cal_dis_mat.sum_cluster_ham_dis_hor(en,count,j,dis_m,hl)
                    caNewmother = caOldmother - hl + dis_m[index][j]
                    for i in range(0,en):
                        A[i][j] = A[i][j]*caOldmother/caNewmother
        #去掉index所在的行与列
        A = common.remove_mat_rc(A,[index,index])
        dis_m = cal_dis_mat.update_dis_mat_del(dis_m,index,index)
        order.remove(hash)
        dic.pop(hash)
    # 针对所有的情况，将大于num_i的所有数据-1
    dic = cal_cluster.dic_index_forward_move(dic,num_i)
    return [A,dic,order,dis_m]


