import time
from utils import Utils
import problem.HLSDSE as HLS
# import HLS.datasets as datasets
import HLS.datasets_DB4HLS as datasets2

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import LatticeTraversingDSE_Python.lattice_exploration_offline as la
import LatticeTraversingDSE_Python.lattice_utils as lattice_utils
from itertools import zip_longest
import copy
import json


class MOEAD:
    # 0表示最小化目标求解，1最大化目标求解。（约定）
    problem_type = 0
    # problem_type=1
    # 测试函数
    Test_fun = HLS
    # 动态展示的时候的title名称
    name = 'HLS'
    # 使用那种方式、DE/GA 作为进化算法
    # GA_DE_Utils = Utils.DE_Utils
    GA_DE_Utils = Utils.GA_Utils
    GA_C_GA1_Utils = Utils.C_GA
    NSGA_Utils = Utils.NSGA

    entire_ds = []

    # 种群大小，取决于vector_csv_file/下的xx.csv
    Pop_size = -1
    # 最大迭代次数
    max_gen = 40
    # 邻居设定（只会对邻居内的相互更新、交叉）
    T_size = 10
    # 支配前沿ID
    EP_X_ID = []
    # 支配前沿 的 函数值
    EP_X_FV = []

    h = 219
    # 种群
    Pop = []
    # 种群计算出的函数值
    Pop_FV = []
    # 权重
    W = []
    # 权重的T个邻居。比如：T=2，(0.1,0.9)的邻居：(0,1)、(0.2,0.8)。永远固定不变
    W_Bi_T = []
    # 理想点。（比如最小化，理想点是趋于0）
    # ps:实验结论：如果你知道你的目标，比如是极小化了，且理想极小值(假设2目标)是[0,0]，
    # 那你就一开始的时候就写死moead.Z=[0,0]吧
    Z = []
    # 权重向量存储目录
    csv_file_path = 'vector_csv_file'
    # 当前迭代代数
    gen = 0
    # 是否动态展示
    need_dynamic = False
    # need_dynamic = TrueH
    # 是否画出权重图
    draw_w = True
    # 用于绘图：当前进化种群中，哪个，被，正在 进化。draw_w=true的时候可见
    now_y = []


    # 调用数据库，引用特征集，配置集
    # benchmark = ["ChenIDCt", "adpcm_decode", "adpcm_encode", "Autocorrelation", "Reflection_coefficients"]
    # benchmark = "Autocorrelation"
    #
    # database = datasets.Datasets(benchmark)
    # synthesis_result = database.benchmark_synthesis_results
    # configurations = database.benchmark_configurations
    # feature_sets = database.benchmark_feature_sets
    # directives = database.benchmark_directives

    # pareto_frontier_exhaustive, pareto_frontier_exhaustive_idx = ADRSunit.pareto_frontier2d(entire_ds)

    # DB4HLS1010
    # benchmark = ['ellpack_ellpack_spmv', 'md_kernel_knn_md', 'viterbi_viterbi_viterbi', 'bbgemm_blocked_gemm', 'merge_merge_sort', 'ms_mergesort_merge_sort',
    # 'get_oracle_activations1_backprop_backprop', 'get_oracle_activations2_backprop_backprop', 'matrix_vector_product_with_bias_input_layer', 'matrix_vector_product_with_bias_second_layer', 'matrix_vector_product_with_bias_output_layer']
    # db = datasets2.Datasets_DB4HLS(name="'aes_addRoundKey_cpy_aes_aes'")
    # synthesis_result = db.benchmark_synthesis_results
    # configurations = db.benchmark_configurations
    # feature_sets = db.benchmark_feature_sets
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
    # entire_ds = []
    # for i in range(0, len(synthesis_result)):
    #     entire_ds.append(DSpoint(synthesis_result[i][0], synthesis_result[i][1], list(configurations[i])))
    #
    # print("in moead:", len([point for point in entire_ds if point.isSynthesis == 1]))
    # entire_ds = []
    # # 更新entire_ds
    # for i in range(0, len(synthesis_result)):
    #     # 存储所有可能的综合点信息
    #     # [i][0]储存latency，[i][1]储存area，用DSpoint存储信息
    #     entire_ds.append(DSpoint(synthesis_result[i][0], synthesis_result[i][1], list(configurations[i])))


    # 得到最大最小值，用于归一化。
    max_latency = max(x[0] for x in synthesis_result)
    min_latency = min(x[0] for x in synthesis_result)
    max_area = max(x[1] for x in synthesis_result)
    min_area = min(x[1] for x in synthesis_result)


    draw_w=True

    def __init__(self, init_samples):
        self.Init_data(init_samples)




    def Init_data(self, init_samples):
        # 加载设计空间
        Utils.Load_entire_ds(self)
        # 加载权重
        Utils.Load_W(self)
        # 计算每个权重Wi的T个邻居
        Utils.cpt_W_Bi_T(self)
        # 创建种群
        self.GA_DE_Utils.Creat_Pop(self, init_samples)
        # 初始化Z集，最小问题0,0
        Utils.cpt_Z(self)



    def show(self):
        if self.draw_w:
            Utils.draw_W(self)
        Utils.draw_MOEAD_Pareto(self, self.name + "num:" + str(self.max_gen) + "")
        Utils.show()

    def run(self):
        initial_pop = copy.deepcopy(self.Pop)
        initial_pop_2 = copy.deepcopy(self.Pop)
        initial_pop_3 = copy.deepcopy(self.Pop)
        start_time = time.time()        # 统计运行时间
        # MOEAD/EDA
        EP_X_ID, adrs_evolution,time_evolution, n_of_synthesis, initial_sampling_size = self.GA_DE_Utils.envolution(self)
        # print(adrs_evolution)
        # time1 = time.time()
        # moead_runtime = time1 - start_time

        # Lattice
        # adrs_evolution_2, n_of_synthesis_2 = la.evolution(self, initial_pop)
        # time2 = time.time()
        # lattice_runtime = time2 - time1

        # E-GA
        # EP_X_ID_3, adrs_evolution_3, n_of_synthesis_3, initial_sampling_size_3 = self.GA_C_GA1_Utils.envolution(self, initial_pop_2)
        # print(adrs_evolution_3)
        # time3 = time.time()
        # E_GA_runtime = time3 - time2

        # NSGA-II
        # adrs_evolution_4 = self.NSGA_Utils.evolution(self, initial_pop_3)
        # time4 = time.time()
        # NSGA_runtime = time4 - time3
        # runtime_list = [moead_runtime, lattice_runtime, E_GA_runtime, NSGA_runtime]
        # # self.show()



        return adrs_evolution, time_evolution, initial_sampling_size




if __name__ == '__main__':
    start_time = time.time()
    benchmark = "get_delta_matrix_weights2"
    filename1 = "backprop"
    Population_size = 220
    with open(f'C:/Users/19519/Desktop/samples/{benchmark}_samples_{Population_size}.json', 'r') as file:
        init_samples = json.load(file)
    n_of_run_times = 10
    adrs_run_history = []
    time_history = []
    # adrs_run_history_2 = []
    # adrs_run_history_3 = []
    # adrs_run_history_4 = []
    runtime_list_history = []
    for i in range(n_of_run_times):
        print(f"====================================Run #{i}次")
        moead = MOEAD(init_samples[i])
        adrs_evolution, time_evolution,initial_sampling_size = moead.run()
        adrs_run_history.append(adrs_evolution)
        time_history.append(time_evolution)
        # adrs_run_history_2.append(adrs_evolution_2)
        # adrs_run_history_3.append(adrs_evolution_3)
        # adrs_run_history_4.append(adrs_evolution_4)
        # runtime_list_history.append(runtime_list)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    with open(f'C:/Users/19519/Desktop/results/EDA/{filename1}/run_time_EDA_{benchmark}.json',
              'w') as f:
        json.dump(elapsed_time, f)
    with open(f'C:/Users/19519/Desktop/results/EDA/{filename1}/EDA_{benchmark}.json', 'w') as f:
        json.dump(adrs_run_history, f)
    with open(f'C:/Users/19519/Desktop/results/EDA/{filename1}/time_EDA_{benchmark}.json', 'w') as f:
        json.dump(time_history, f)


    # 画图部分：
    averages_adrs = list(list(map(lattice_utils.avg, lattice_utils.zip_longest_fill_last(*adrs_run_history))))
    # averages_adrs_2 = list(list(map(lattice_utils.avg, lattice_utils.zip_longest_fill_last(*adrs_run_history_2))))
    # averages_adrs_3 = list(list(map(lattice_utils.avg, lattice_utils.zip_longest_fill_last(*adrs_run_history_3))))
    # averages_adrs_4 = list(list(map(lattice_utils.avg, lattice_utils.zip_longest_fill_last(*adrs_run_history_4))))
    plt.title("ADRS evolution")
    plt.ylabel("mean ADRS")
    plt.xlabel("# of synthesis")
    plt.plot(range(Population_size, len(averages_adrs)+Population_size), averages_adrs, label='MOEAD/EDA')
    # plt.plot(range(initial_sampling_size, len(averages_adrs_2)+initial_sampling_size), averages_adrs_2, label='Lattice')
    # plt.plot(range(initial_sampling_size, len(averages_adrs_3)+initial_sampling_size), averages_adrs_3, label='ξ-Constraint GA')
    # plt.plot(range(initial_sampling_size, len(averages_adrs_4)+initial_sampling_size), averages_adrs_4, label='NSGA-Ⅱ')
    plt.legend()
    plt.grid()
    plt.savefig('ADRS evolution.png')
    plt.show()

    plt.show()

    # 统计adrs平均变化
    print("moead/eda average_adrs_evolution: ", averages_adrs)
    # print("lattice average_adrs_evolution: ", averages_adrs_2)
    # print("e-ga average_adrs_evolution: ", averages_adrs_3)
    # print("NSGA-II average_adrs_evolution: ", averages_adrs_4)
    # 统计adrs小于0.04时的综合次数
    idx = next((i for i, v in enumerate(averages_adrs) if v < 0.04), None)
    # idx_2 = next((i for i, v in enumerate(averages_adrs_2) if v < 0.04), None)
    # idx_3 = next((i for i, v in enumerate(averages_adrs_3) if v < 0.04), None)
    # idx_4 = next((i for i, v in enumerate(averages_adrs_4) if v < 0.04), None)

    if idx is not None:
        syn_nums1 = idx + Population_size
        print("moead/eda adrs<0.04:", syn_nums1)
    else:
        print("moead/eda adrs<0.04: None")

    # if idx_2 is not None:
    #     syn_nums2 = idx_2 + initial_sampling_size
    #     print("lattice adrs<0.04:", syn_nums2)
    # else:
    #     print("lattice adrs<0.04: None")
    #
    # if idx_3 is not None:
    #     syn_nums3 = idx_3 + initial_sampling_size
    #     print("e-ga adrs<0.04:", syn_nums3)
    # else:
    #     print("e-ga adrs<0.04: None")
    #
    # if idx_4 is not None:
    #     syn_nums4 = idx_4 + initial_sampling_size
    #     print("NSGA-II adrs<0.04:", syn_nums4)
    # else:
    #     print("NSGA-II adrs<0.04: None")
    # 统计最终adrs值大小
    final_adrs = []
    final_adrs.append(averages_adrs[-1])
    # final_adrs.append(averages_adrs_2[-1])
    # final_adrs.append(averages_adrs_3[-1])
    # final_adrs.append(averages_adrs_4[-1])
    print("the final adrs is :", final_adrs)
    # 统计运行时间
    average_runtime = [sum(values)/len(values) for values in zip(*runtime_list_history)]
    print("the runtime is :", average_runtime)




