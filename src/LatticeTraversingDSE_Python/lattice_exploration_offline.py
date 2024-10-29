from lattice_data import Lattice
from lattice_synthesis import VivdoHLS_Synthesis
from lattice_synthesis import FakeSynthesis
from lattice_ds_point import DSpoint
from lattice_sphere_tree import SphereTree as st
import lattice_utils
import datasets
import numpy as np
from itertools import zip_longest
import copy
from matplotlib import pyplot as plt
import sys
import pandas as pd

sys.setrecursionlimit(15200)
# sys.setrecursionlimit(10000)
np.random.seed(42)

# Set the radius for the exploration
# 设置最大distance
radius = 0.5
# radius = 1

# Set number of runs due to the probabilistic initial sampling
# 运行次数
n_of_runs = 10

# To plot the ADRS chart
plot = True

# Initial sampling size
# 初始采样大小
intial_sampling_size = 80

# Performance goal
pf_goal = 0

# Collect stats
# 收集数据
max_n_of_synth = 0
adrs_run_stats = []
goal_stats_history = []
goal_stats = []

# Read dataset

# Specify the list of dataset to explore
# 选择benchmark

def evolution(moead, pop):
    # Extract data from the database
    # 提取数据： 综合结果、configurations 、特征集、指令。
    synthesis_result = moead.synthesis_result
    configurations = moead.configurations
    feature_sets = moead.feature_sets
    entire_ds = moead.entire_ds

    # Set variables to store exploration data
    # 存储探索数据
    adrs_run_history = []
    run_stats = []
    # 遍历每次run
    goal_acheived = False
    # Collect stats
    online_statistics = {}
    online_statistics['adrs'] = []
    online_statistics['delta_adrs'] = []
    online_statistics['n_synthesis'] = []
    # run_stats = None

    # 探索球元素
    sphere_elements_sizes = []
    # 设置最大半径
    max_radius = 0.5
    # max_radius = 1
    # Create Lattice
    # 创建lattice
    lattice = Lattice(feature_sets, radius)

    # Generate inital samples
    # 产生初始采样 采用同样的初始种群
    initial_samples = pd.Series(pop).drop_duplicates().tolist()       # 未离散化
    samples = []
    for s in initial_samples:
        c = lattice.revert_original_config(s)
        samples.append(c)

    # samples = samples_dataset
    # 记录采样大小
    n_of_synthesis = len(samples)

    # Synthesise sampled configuration
    # FakeSynthesis simulates the synthesis process retrieving the configuration from the proper DB
    # 调用hls对象 完成后续综合操作
    hls = FakeSynthesis(entire_ds, lattice)
    # 综合结果收集
    sampled_configurations_synthesised = []
    # 遍历每个采样点
    for s in samples:
        # 得到s的latency, area
        latency, area = hls.synthesise_configuration(s)
        # 存入到DSpoint对象 存储合成点信息
        synthesised_configuration = DSpoint(latency, area, s)
        # 放入到采样列表中
        sampled_configurations_synthesised.append(synthesised_configuration)
        # 放入格子空间
        lattice.lattice.add_config(s)
    # sampled_configurations_synthesised列表收集完所有初始采样点的latency，area，configuration信息，开始寻找帕累托边界
    # After the inital sampling, retrieve the pareto frontier
    pareto_frontier, pareto_frontier_idx = lattice_utils.pareto_frontier2d(sampled_configurations_synthesised)

    # Get exhaustive pareto frontier (known only if ground truth exists)
    pareto_frontier_exhaustive, pareto_frontier_exhaustive_idx = lattice_utils.pareto_frontier2d(entire_ds)

    # Store a copy to save the pareto front before the exploration algorithm
    pareto_frontier_before_exploration = copy.deepcopy(pareto_frontier)

    # Calculate ADRS
    # ADRS进化过程
    adrs_evolution = []
    # 计算adrs
    adrs = lattice_utils.adrs2d(pareto_frontier_exhaustive, pareto_frontier)
    # ADRS after initial sampling
    adrs_evolution.append(adrs)

    # Select randomly a pareto configuration and explore its neighbourhood
    # 随机选择一个帕累托结点进行领域结点探索
    r = np.random.randint(0, len(pareto_frontier))
    # 得到所有帕累托配置
    pareto_configurations = [samples[i] for i in pareto_frontier_idx]
    # 选出索引为r的帕累托配置
    pareto_solution_to_explore = pareto_configurations[r]

    # Search locally for the configuration to explore
    # 对此点进行局部搜索，调用st对象
    sphere = st(pareto_solution_to_explore, lattice)
    # 选出离此帕累托结点最近的配置
    new_configuration = sphere.random_closest_element

    # Until there are configurations to explore, try to explore these
    while new_configuration is not None:
        # Synthesise configuration
        # 合成此结点，得到其信息存入DS结点
        latency, area = hls.synthesise_configuration(new_configuration)
        # Generate a new design point
        ds_point = DSpoint(latency, area, new_configuration)

        # Update known synthesis values and configurations(only pareto + the new one)
        pareto_frontier.append(ds_point)

        # Add configuration to the tree
        lattice.lattice.add_config(ds_point.configuration)

        # Get pareto frontier
        # 更新新的帕累托边界
        pareto_frontier, pareto_frontier_idx = lattice_utils.pareto_frontier2d(pareto_frontier)

        # Calculate ADRS
        # 计算新的adrs，并加入到adrs_evolution列表
        adrs = lattice_utils.adrs2d(pareto_frontier_exhaustive, pareto_frontier)
        adrs_evolution.append(adrs)

        # Find new configuration to explore
        # Select randomly a pareto configuration
        # 继续随机探索下一个
        search_among_pareto = copy.copy(pareto_frontier)
        while len(search_among_pareto) > 0:
            r = np.random.randint(0, len(search_among_pareto))
            pareto_solution_to_explore = search_among_pareto[r].configuration

            # Explore the closer element locally
            sphere = st(pareto_solution_to_explore, lattice)
            # 得到new_configuration用于下一轮while循环探索
            new_configuration = sphere.random_closest_element
            # 条件判断，判断是否需要弹出当前帕累托结点
            if new_configuration is None:
                search_among_pareto.pop(r)
                continue

            max_radius = max(max_radius, sphere.radius)
            if max_radius > lattice.max_distance:
                search_among_pareto.pop(r)
                continue

            break

        exit_expl = False
        if len(search_among_pareto) == 0:
            print("Exploration terminated")
            exit_expl = True

        if max_radius > lattice.max_distance:
            print("Max radius reached")
            exit_expl = True

        # Here eventually add a condition to limit the number of synthesis to a certain treshold
        # if n_of_syntheis == budget:
        #     exit_expl = True


        if adrs <= pf_goal and not goal_acheived:
            # adrs小于设定目标且goal_acheived处于false状态
            # 说明以达到合成条件，添加达到goal所需的综合次数
            goal_acheived = True
            goal_stats.append(n_of_synthesis)
            exit_expl = True

        # if max_n_of_synth < n_of_synthesis:
        #     max_n_of_synth = n_of_synthesis

        n_of_synthesis += 1
        # 将adrs信息以及n_of_synthesis数量放入到online_statistics字典中
        run_stats = lattice_utils.collect_online_statis(online_statistics, adrs, n_of_synthesis)

        # 如果到了退出点
        if exit_expl:
            # If the exploration is ending update the calculate final ADRS
            # 计算最终的adrs并更新adrs_evolution
            adrs = lattice_utils.adrs2d(pareto_frontier_exhaustive, pareto_frontier)
            adrs_evolution.append(adrs)
            print("Number of synthesis:\t{:d}".format(n_of_synthesis))
            print("Max radius:\t{:0.4f}".format(max_radius))
            print("Final ADRS:\t{:0.4f}".format(adrs))
            print()
            # print(lattice.lattice.get_n_of_children())
            # goal_stats.append(n_of_synthesis)
            break
    return adrs_evolution, n_of_synthesis


    # adrs_run_stats.append(run_stats)
    # adrs_run_history.append(adrs_evolution)
    # goal_stats_history.append(goal_stats)

# collect_offline_stats = lattice_utils.collect_offline_stats(adrs_run_stats, n_of_runs, max_n_of_synth, goal_stats)








