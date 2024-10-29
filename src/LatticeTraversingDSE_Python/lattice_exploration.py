from lattice_data import Lattice
from lattice_synthesis import VivdoHLS_Synthesis
from lattice_ds_point import DSpoint
from lattice_sphere_tree import SphereTree as st
import lattice_utils
import datasets
# import matplotlib.pyplot as plt
import numpy as np
import copy

Autocorrelation_extended = datasets.Datasets("Autocorrelation_extended")
# 获得特征集
feature_sets = [i[1] for i in Autocorrelation_extended.autcorrelation_extended_directives_ordered]

# While used to run the experiments multiple times
# 只在第一次进行画图
n_of_runs = 1
if n_of_runs > 1:
    plot_chart = False
else:
    plot_chart = True

collected_run = []
# 遍历每一次运行
for run in range(n_of_runs):
    # Create Lattice 创造格子空间 最大distance设置为10
    lattice = Lattice(feature_sets, 10)
    max_radius = 0

    # Probabilistic sample according to beta distribution
    # samples = lattice.beta_sampling(0.1, 0.1, 20)

    # Populate the tree with the initial sampled values
    # lattice.lattice.populate_tree(samples)
    # n_of_synthesis = len(samples)

    # Synthesise sampled configuration
    # hls = FakeSynthesis(entire_ds, lattice)
    prj_description = {"prj_name": "Autocorrelation_extended",
                       "test_bench_file": "gsm.c",
                       "source_folder": "<path_to_src_folder>",
                       "top_function": "Autocorrelation"}

    # 创建综合对象，方便后续综合操作   这个prj_description好像有点问题
    hls = VivdoHLS_Synthesis(lattice, Autocorrelation_extended.autcorrelation_extended,
                             Autocorrelation_extended.autcorrelation_extended_directives_ordered,
                             Autocorrelation_extended.autcorrelation_extended_bundling,
                             prj_description)
    # sampled_configurations_synthesised列表
    sampled_configurations_synthesised = []
    # for s in samples:
    samples = []
    # 初始采样点为20个
    while len(samples) < 20:
        # 得到一个sample
        sample = lattice.beta_sampling(0.1, 0.1, 1).pop()
        # 得到对应sample的latency和area
        latency, area = hls.synthesise_configuration(sample)
        # if latency is None:
        #     lattice.lattice.add_config(sample)
        #     continue
        # 将此点加入到simple列表中
        samples.append(sample)
        # 创建合成点对象（存储了半径，latency，area，configuration变量信息）
        synthesised_configuration = DSpoint(latency, area, sample)
        # 添加到sampled_configurations_synthesised列表中
        sampled_configurations_synthesised.append(synthesised_configuration)
        # 在格子空间中添加配置
        lattice.lattice.add_config(sample)
    # --------------------------------------------------------
    # 初始采样完毕
    # 得到合成点的数量
    n_of_synthesis = len(samples)
    print(samples)
    print(len(samples))
    print(len(sampled_configurations_synthesised))
    # Get pareto frontier from sampled configuration
    # 从初始采样点中得到帕累托边界以及对应索引
    pareto_frontier, pareto_frontier_idx = lattice_utils.pareto_frontier2d(sampled_configurations_synthesised)

    # Get exhaustive pareto frontier (known only if ground truth exists)
    # pareto_frontier_exhaustive, pareto_frontier_exhaustive_idx = lattice_utils.pareto_frontier2d(entire_ds)

    # 探索前帕累托边界，并创建初始的latency和area列表
    pareto_frontier_before_exploration = copy.deepcopy(pareto_frontier)
    intial_pareto_frontier_latency = []
    intial_pareto_frontier_area = []
    for pp in pareto_frontier_before_exploration:
        intial_pareto_frontier_latency.append(pp.latency)
        intial_pareto_frontier_area.append(pp.area)

    # # PLOT start
    # if plot_chart:
    #     for p in sampled_configurations_synthesised:
    #         plt.scatter(p.latency, p.area, color='b')
    #
    #     for pp in pareto_frontier_exhaustive:
    #         plt.scatter(pp.latency, pp.area, color='r')
    #
    #     for pp in pareto_frontier:
    #         plt.scatter(pp.latency, pp.area, color='g')
    #
    #     plt.grid()
    #     # plt.draw()
    #     # PLOT end

    # Calculate ADRS
    # ADRS的进化过程
    adrs_evolution = []
    # adrs = lattice_utils.adrs2d(pareto_frontier_exhaustive, pareto_frontier)
    # 将初始采样得到的adrs添加到adrs_evolution的第一项
    adrs = lattice_utils.adrs2d(pareto_frontier_before_exploration, pareto_frontier)
    adrs_evolution.append(adrs)

    # Select randomly a pareto configuration and find explore his neighbour
    # 随机选择一个pareto configuration
    r = np.random.randint(0, len(pareto_frontier))
    # 生成一个帕累托配置列表
    pareto_configurations = [samples[i] for i in pareto_frontier_idx]
    # 探索 索引为r（随机得到）的帕累托结点
    configuration_to_explore = pareto_configurations[r]

    # Search locally for the configuration to explore

    # 生成shpere_tree对象进行后续local search
    sphere = st(configuration_to_explore, lattice)
    # 选择搜索球中最近的元素进行合成
    new_configuration = sphere.random_closest_element

    # Until there are configurations to explore, try to explore these
    while new_configuration is not None:
        print("New iteration")
        # Synthesise configuration
        # 得到新配置的latency和area
        latency, area = hls.synthesise_configuration(new_configuration)
        # if latency is None:
        #     lattice.lattice.add_config(new_configuration)
        #     # Find new configuration to explore
        #     # Select randomly a pareto configuration
        #     r = np.random.randint(0, len(pareto_frontier))
        #     pareto_solution_to_explore = pareto_frontier[r].configuration
        #
        #     # Explore the closer element locally
        #     sphere = st(pareto_solution_to_explore, lattice)
        #     new_configuration = sphere.random_closest_element
        #     max_radius = max(max_radius, sphere.radius)
        #
        #     if new_configuration is None:
        #         print "Exploration terminated"
        #         break
        #     if max_radius > lattice.max_distance:
        #         print "Exploration terminated, max radius reached"
        #         break
        #     continue
        # Generate a new design point
        ds_point = DSpoint(latency, area, new_configuration)
        print( "Lat:", latency, "\tArea:", area)

        # Update known synthesis values and configurations(only pareto + the new one)
        # 加入到pareto_frontier列表中，方便后续比较更新出新的pareto_frontier
        pareto_frontier.append(ds_point)

        # Add configuration to the tree
        # 加入到合成树中
        lattice.lattice.add_config(ds_point.configuration)

        # Get pareto frontier
        # 更新新的pareto_frontier
        pareto_frontier, pareto_frontier_idx = lattice_utils.pareto_frontier2d(pareto_frontier)

        # Calculate ADRS
        # adrs = lattice_utils.adrs2d(pareto_frontier_exhaustive, pareto_frontier)
        # 更新新的adrs
        adrs = lattice_utils.adrs2d(pareto_frontier_before_exploration, pareto_frontier)
        # 更新 adrs_evolution 列表
        adrs_evolution.append(adrs)
    #    if adrs == 0:
    #         break

        # Find new configuration to explore
        # 再次从帕累托结点集中探索新的configuration
        search_among_pareto = copy.copy(pareto_frontier)
        while len(search_among_pareto) > 0:
            r = np.random.randint(0, len(search_among_pareto))
            # 随机得到一个帕累托结点进行探索
            pareto_solution_to_explore = search_among_pareto[r].configuration
            # Explore the closer element locally
            sphere = st(pareto_solution_to_explore, lattice)
            # 生成一个new_configuration用以下一次循环
            new_configuration = sphere.random_closest_element
            if new_configuration is None:
                # 如果该结点已找不到最近的configuration，弹出该帕累托结点，跳出循环，找其他帕累托结点的邻近configuration
                search_among_pareto.pop(r)
                continue

            # shpere 树在每次探索中，如果在半径内已经找不到最近的元素，每次会增加一个min_increment 因此需要做此判断
            max_radius = max(max_radius, sphere.radius)
            if max_radius > lattice.max_distance:
                # 如果已经超过最大探索半径，弹出该点。
                search_among_pareto.pop(r)
                continue

            break

        exit_expl = False
        # search_among_pareto = 0时，所有帕累托点的邻近结点被探索，探索结束。
        if len(search_among_pareto) == 0:
            print( "Exploration terminated")
            exit_expl = True

        # 达到最大半径，探索退出
        if max_radius > lattice.max_distance:
            print("Max radius reached")
            exit_expl = True

        # 如果 exit_expl = True 跳出循环
        if exit_expl:
            break
        # 否则合成点+1
        n_of_synthesis += 1
        print(n_of_synthesis)

    # 得到最终的帕累托边界的latency和area信息
    final_pareto_frontier_latency = []
    final_pareto_frontier_area = []
    for pp in pareto_frontier:
        final_pareto_frontier_latency.append(pp.latency)
        final_pareto_frontier_area.append(pp.area)

    # 收集此次 run 的合成点数量，adrs_evolution，max_radius
    collected_run.append((n_of_synthesis, adrs_evolution, max_radius))
    # 收集完毕，重置
    n_of_synthesis = 0
    adrs_evolution = []
    max_radius = 0

    # if plot_chart:
    #     fig1 = plt.figure()
    #     for p in sampled_configurations_synthesised:
    #         plt.scatter(p.latency, p.area, color='b')
    #
    #     for pp in pareto_frontier_exhaustive:
    #         plt.scatter(pp.latency, pp.area, color='r', s=40)
    #
    #     for pp in pareto_frontier:
    #         plt.scatter(pp.latency, pp.area, color='g')
    #
    #     fig2 = plt.figure()
    #     plt.grid()
    #     pareto_frontier.sort(key=lambda x: x.latency)
    #     plt.step([i.latency for i in pareto_frontier], [i.area for i in pareto_frontier], where='post', color='r')
    #     pareto_frontier_before_exploration.sort(key=lambda x: x.latency)
    #     plt.step([i.latency for i in pareto_frontier_before_exploration], [i.area for i in pareto_frontier_before_exploration], where='post', color='b')
    #     # plt.draw()
    #
    #     fig3 = plt.figure()
    #     plt.grid()
    #     plt.plot(adrs_evolution)
    #     plt.show()


#mean_adrs, radii, final_adrs_mean = lattice_utils.get_statistics(collected_run)
mean_adrs, radii, final_adrs_mean = lattice_utils.collect_online_statis(collected_run)
data_file = open("mean_adrs.txt", "w")
data_file.write(str(mean_adrs))
data_file.close()

data_file = open("radii.txt", "w")
data_file.write(str(radii))
data_file.close()

data_file = open("final_adrs_mean.txt", "w")
data_file.write(str(final_adrs_mean))
data_file.close()

data_file = open("inital_pareto.txt","w")
data_file.write(str(intial_pareto_frontier_latency))
data_file.write(str(intial_pareto_frontier_area))
data_file.close()

data_file = open("final_pareto.txt","w")
data_file.write(str(final_pareto_frontier_latency))
data_file.write(str(final_pareto_frontier_area))
data_file.close()
# print mean_adrs
# plt.plot(mean_adrs)
# plt.show()
