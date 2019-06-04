# coding=utf-8
# 计算距离矩阵A的函数
import numpy as np
#from . import common
import common


# 对digit数据进行上三角距离计算Hamming
def cal_dis_mat(list, method, hl, th, oper, isRestrict=False):
    dig_len = len(list)
    dis_matrix_origin = np.zeros((dig_len, dig_len))
    dis_matrix = np.zeros((dig_len, dig_len))
    for indexR, itemR in enumerate(list):
        cursor = indexR + 1
        dis_matrix_origin[indexR][indexR] = 0
        while cursor < dig_len:
            dis_matrix_origin[indexR][cursor] = common.cal_elements_dis(itemR, list[cursor], method=method)
            dis_matrix[indexR][cursor] = dis_matrix_origin[indexR][cursor]
            if isRestrict == True:
                dis_matrix[indexR][cursor] = common.revalue_distance(dis_matrix[indexR][cursor], th, oper)
            dis_matrix_origin[cursor][indexR] = dis_matrix_origin[indexR][cursor]
            dis_matrix[cursor][indexR] = dis_matrix[indexR][cursor]
            cursor = cursor + 1
    # 再一次对dis_matrix进行调整，避免孤立节点
    # print("原始距离矩阵：\n", dis_matrix_origin)
    # print("restrict距离矩阵：\n", dis_matrix)
    # dict_value, dict_index = record_mindis(dis_matrix_origin, dig_len, hl)
    # print("最小距离：\n", dict_value, "\n", "索引：\n", dict_index)
    # dis_matrix_sum = dis_matrix.sum(0)
    # for i in range(dig_len):
    #     if dis_matrix_sum[i] == 1 - dig_len:
    #         for j in dict_index[i]:
    #             dis_matrix[i][j] = dict_value[i]
    #             dis_matrix[j][i] = dict_value[i]
    # print("避免孤立节点后的距离矩阵：\n", dis_matrix)
    return dis_matrix


# 记录和每个节点距离最近的节点索引及距离
def record_mindis(dis_mat, nodes_num, hl):
    mindis_dict_value = {}
    mindis_dict_index = {}
    min_value = hl
    for i in range(0, nodes_num):
        index_list = []
        for j in range(0, nodes_num):
            if i != j and dis_mat[i][j] < min_value:
                min_value = dis_mat[i][j]
                index_list.clear()
                index_list.append(j)
            elif i != j and dis_mat[i][j] == min_value:
                index_list.append(j)
        mindis_dict_value[i] = min_value
        mindis_dict_index[i] = index_list
        min_value = hl
    # mindis_dict_value = {1:24,2:13,3:34,...}
    # mindis_dict_index = {1:[2],2:[4,6],...}
    return mindis_dict_value, mindis_dict_index


# 更新距离矩阵 np.r_给矩阵加上一行，np.c_给矩阵加上一列
def update_dis_mat_add(order, hash, org_dis_m, th, oper, dis_method, isRestrict):
    en = len(order)
    A_line = np.zeros(en)
    for i in range(0, en):
        A_line[i] = common.cal_elements_dis(order[i], hash, dis_method)
        if isRestrict == True:
            A_line[i] = common.revalue_distance(A_line[i], th, oper)
    return common.extend_mat_rc(org_dis_m, A_line, A_line, 0)


# 更新距离矩阵，删除某行某列
def update_dis_mat_del(dis_m, index_r, index_c):
    return common.remove_mat_rc(dis_m, [index_r, index_c])


# 单点情况下计算每一列的汉明距离和
def sum_node_ham_dis_hor(en, j, dis_m, hl):
    return common.sum_node_mat_hor(en, j, dis_m, hl)


# 团簇条件下按照单点统计汉明距离下列和
def sum_cluster_ham_dis_hor(en, order_count, j, dis_m, hl):
    return common.sum_cluster_mat_hor(en, order_count, j, dis_m, hl)
