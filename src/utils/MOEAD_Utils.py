import os
from math import sqrt
from src.HLS.DSpoint import DSpoint
import src.HLS.ADRSunit as ADRSunit
import src.HLS.datasets as datasets
import src.HLS.datasets_DB4HLS as datasets2

import numpy as np
from src.utils.Mean_Vector_Util import Mean_vector

'''
MOEAD工具包
'''

def Load_entire_ds(moead):
    # # benchmark = ["ChenIDCt", "adpcm_decode", "adpcm_encode", "Autocorrelation", "Reflection_coefficients"]
    # benchmark = "Autocorrelation"
    #
    # database = datasets.Datasets(benchmark)
    # synthesis_result = database.benchmark_synthesis_results
    # configurations = database.benchmark_configurations
    #
    # DB4HLS
    # benchmark = ['ellpack_ellpack_spmv', 'md_kernel_knn_md', 'viterbi_viterbi_viterbi', 'bbgemm_blocked_gemm', 'merge_merge_sort', 'ms_mergesort_merge_sort',
    # 'get_delta_matrix_weights1'. 'get_oracle_activations1_backprop_backprop', 'get_oracle_activations2_backprop_backprop', 'matrix_vector_product_with_bias_input_layer'
    # 'matrix_vector_product_with_bias_second_layer', 'matrix_vector_product_with_bias_output_layer']
    benchmark = "get_delta_matrix_weights2"
    database = datasets2.Datasets(benchmark)
    synthesis_result = database.benchmark_synthesis_results
    synthesis_result = [
        ('999999999', '999999999', '0')
        if '0' in item or any(val is None for val in item)
        else item
        for item in synthesis_result
    ]
    configurations = database.benchmark_configurations
    feature_sets = database.benchmark_feature_sets
    synthesis_result = [(float(x), float(y), float(z)) for x, y, z in synthesis_result]

    # 创建设计空间
    entire_ds = []
    for i in range(0, len(synthesis_result)):
        entire_ds.append(
            DSpoint(synthesis_result[i][0], synthesis_result[i][1],  synthesis_result[i][2], list(configurations[i])))

    moead.entire_ds = entire_ds

    return entire_ds

def Load_W(moead):
    file = moead.name + '.csv'
    path = moead.csv_file_path + '/' + file
    if os.path.exists(path) == False:
        print('not exists')
        mv = Mean_vector(moead.h, moead.Test_fun.Func_num, path)
        mv.generate()
        print('created')
    W = np.loadtxt(fname=path)
    moead.Pop_size = W.shape[0]
    moead.W = W
    return W


def cpt_Z(moead):
    # 初始化Z集，最小问题0,0，..。
    # ps:实验结论：如果你知道你的目标，比如是极小化了，且理想极小值(假设2目标)是[0,0]，
    # 那你就一开始的时候就写死moead.Z=[0,0]吧，就不用这个函数进行设置了。
    # 这里极小化全部初始化为[0,0,...0]，极大化：[10,10,....10]
    Z = []
    for fi in range(moead.Test_fun.Func_num):
        z_i = -1
        if moead.problem_type == 0:
            z_i = 0
        if moead.problem_type == 1:
            z_i = 10
        Z.append(z_i)
    moead.Z = Z
    return Z


def cpt_Z2(moead):
    # 初始化Z集，最小问题0,0，..
    # 第一个个体的所有目标函数值
    Z = moead.Pop_FV[0][:]
    dz = np.random.rand()
    for fi in range(moead.Test_fun.Func_num):   # 遍历每一个目标函数
        for Fpi in moead.Pop_FV:    # 遍历种群中的每一个个体
            if moead.problem_type == 0:     # 最小化问题
                if Fpi[fi] < Z[fi]:
                    Z[fi] = Fpi[fi] - dz    # 更新
            if moead.problem_type == 1:
                if Fpi[fi] > Z[fi]:
                    Z[fi] = Fpi[fi] + dz
    moead.Z = Z
    return Z


# 计算初始化前沿
def init_EP(moead):
    for pi in range(moead.Pop_size):    # 遍历每一个个体
        np = 0
        F_V_P = moead.Pop_FV[pi]    # 当前个体的目标函数值
        for ppi in range(moead.Pop_size):
            F_V_PP = moead.Pop_FV[ppi]
            if pi != ppi:
                if is_dominate(moead, F_V_PP, F_V_P):
                    np += 1
        if np == 0:
            moead.EP_X_ID.append(pi)
            moead.EP_X_FV.append(F_V_P[:])


# 计算T个邻居
def cpt_W_Bi_T(moead):
    # 计算权重的T个邻居
    if moead.T_size < 1:
        return -1
    for bi in range(moead.W.shape[0]):      # 遍历权重向量的每一行
        Bi = moead.W[bi]
        # 当前权重向量与其他所有权重向量的欧氏距离
        DIS = np.sum((moead.W - Bi) ** 2, axis=1)
        B_T = np.argsort(DIS)   # 按照距离进行排序
        # 第0个是自己（距离永远最小） 选出最近的T个权重
        B_T = B_T[1:moead.T_size + 1]
        moead.W_Bi_T.append(B_T)


def is_dominate(moead, F_X, F_Y):
    entire_ds = moead.entire_ds
    # 判断F_X是否支配F_Y
    if type(F_Y) != list:
        F_X = moead.Test_fun.Func(F_X, entire_ds)
        F_Y = moead.Test_fun.Func(F_Y, entire_ds)
    i = 0
    if moead.problem_type == 0:  # minimize
        for xv, yv in zip(F_X, F_Y):
            if xv < yv:
                i = i + 1
            if xv > yv:
                return False
    if moead.problem_type == 1:  # maximize
        for xv, yv in zip(F_X, F_Y):
            if xv > yv:
                i = i + 1
            if xv < yv:
                return False
    if i != 0:
        return True
    return False


def cpt_to_Z_dist(moead, X):
    entire_ds = moead.entire_ds
    #  计算X点到参考点（Z）的距离
    F_X = moead.Test_fun.Func(X, entire_ds)
    d = 0
    for i, fm in enumerate(F_X):
        d = d + (fm - moead.Z[i]) ** 2
    d = sqrt(d)
    return d


def Tchebycheff_dist(w, f, z):
    # 计算切比雪夫距离
    return w * abs(f - z)


def cpt_tchbycheff(moead, idx, X):
    # idx：X在种群中的位置
    # 计算X的切比雪夫距离（与理想点Z的）
    # 选出最大距离的点
    max = moead.Z[0]
    ri = moead.W[idx]
    F_X = moead.Test_fun.Func(X, moead)
    for i in range(moead.Test_fun.Func_num):
        fi = Tchebycheff_dist(ri[i], F_X[i], moead.Z[i])
        if fi > max:
            max = fi
    return max


def update_BTX(moead, P_B, Y):
    # 根据Y更新P_B集内邻居
    for j in P_B:
        Xj = moead.Pop[j]
        d_x = cpt_tchbycheff(moead, j, Xj)
        d_y = cpt_tchbycheff(moead, j, Y)
        if d_y <= d_x:
            # d_y 的切比雪夫距离更小
            moead.Pop[j] = Y[:]
            F_Y = moead.Test_fun.Func(Y, moead)
            moead.Pop_FV[j] = F_Y
            update_EP_By_ID(moead, j, F_Y)


def update_EP_By_ID(moead, id, F_Y):
    # 如果id存在，则更新其对应函数集合的值
    if id in moead.EP_X_ID:
        # 拿到所在位置
        position_pi = moead.EP_X_ID.index(id)
        # 更新函数值
        moead.EP_X_FV[position_pi][:] = F_Y[:]


def update_Z(moead, Y):
    # 根据Y更新Z坐标。。ps:实验结论：如果你知道你的目标，比如是极小化了，且理想极小值(假设2目标)是[0,0]，
    # 那你就一开始的时候就写死moead.Z=[0,0]把
    dz = np.random.rand()
    F_y = moead.Test_fun.Func(Y, moead)
    for j in range(moead.Test_fun.Func_num):
        if moead.problem_type == 0:  # minimize
            if moead.Z[j] > F_y[j]:
                moead.Z[j] = F_y[j] - dz
        if moead.problem_type == 1:  # maximize
            if moead.Z[j] < F_y[j]:
                moead.Z[j] = F_y[j] + dz


def update_EP_By_Y(moead, id_Y):
    # 根据Y更新前沿
    # 根据Y更新EP
    i = 0
    # 拿到id_Y的函数值
    F_Y = moead.Pop_FV[id_Y]
    # 需要被删除的集合
    Delet_set = []
    # 支配前沿集合，的数量
    Len = len(moead.EP_X_FV)
    for pi in range(Len):   # 遍历EP中的每个个体
        # F_Y是否支配pi号个体，支配？哪pi就完了，被剔除。。
        if is_dominate(moead, F_Y, moead.EP_X_FV[pi]):
            # 列入被删除的集合
            Delet_set.append(pi)
            break
        if i != 0:
            break
        if is_dominate(moead, moead.EP_X_FV[pi], F_Y):
            # 它有被别人支配！！记下来能支配它的个数
            i += 1
    # 新的支配前沿的ID集合，种群个体ID，
    new_EP_X_ID = []
    # 新的支配前沿集合的函数值
    new_EP_X_FV = []
    for save_id in range(Len):
        if save_id not in Delet_set:
            # 不需要被删除，那就保存
            new_EP_X_ID.append(moead.EP_X_ID[save_id])
            new_EP_X_FV.append(moead.EP_X_FV[save_id])
    # 更新上面计算好的新的支配前沿
    moead.EP_X_ID = new_EP_X_ID
    moead.EP_X_FV = new_EP_X_FV
    # 如果i==0，意味着没人支配id_Y
    # 没人支配id_Y？太好了，加进支配前沿呗
    if i == 0:
        # 不在里面直接加新成员
        if id_Y not in moead.EP_X_ID:
            moead.EP_X_ID.append(id_Y)
            moead.EP_X_FV.append(F_Y)
        else:
            # 本来就在里面的，更新它
            idy = moead.EP_X_ID.index(id_Y)
            moead.EP_X_FV[idy] = F_Y[:]
    # over
    return moead.EP_X_ID, moead.EP_X_FV

def check_isSynthesis(moead, p1):
    entire_ds = moead.entire_ds
    for point in entire_ds:
        if point.configuration == p1:
            if point.isSynthesis == 1:
                return True
            else:
                return False
    return False

def adrs_undate(moead, config, parato_frontier, pareto_frontier_exhaustive, n_of_synthesis, adrs_evolution,time_evolution, all_time):
    latency, area, time = moead.Test_fun.latency(config, moead.entire_ds), moead.Test_fun.area(config, moead.entire_ds), moead.Test_fun.time(config, moead.entire_ds)
    ds_point = DSpoint(latency, area, time,config)
    parato_frontier.append(ds_point)
    parato_frontier, parato_frontier_idx = ADRSunit.pareto_frontier2d(parato_frontier)
    adrs = ADRSunit.adrs2d(pareto_frontier_exhaustive, parato_frontier)
    adrs_evolution.append(adrs)
    all_time = time_evolution[-1] + time
    time_evolution.append(all_time)
    n_of_synthesis += 1