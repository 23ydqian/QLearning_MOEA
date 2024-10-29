import copy
import random
import numpy as np
import datasets
from HLS.sythesis import FakeSynthesis
from HLS.DSpoint import DSpoint
import HLS.ADRSunit as ADRSunit
import math
import pandas as pd


# 初始为无穷大


def area_1(config,entire_ds):
    # 引入指令、configuration、area、latency
    hls = FakeSynthesis(entire_ds)
    area = hls.synthesise_configuration(config)[1]
    return area

def latency_1(config,entire_ds):
    # 引入指令、configuration、area、latency
    hls = FakeSynthesis(entire_ds)
    latency = hls.synthesise_configuration(config)[0]
    return latency

def mutate3(moead, solution):
    feature_sets = moead.feature_sets
    mutation_point1, mutation_point2 = random.sample(range(0, len(solution) - 1), 2)
    new_value_1 = random.choice(feature_sets[mutation_point1])
    new_value_2 = random.choice(feature_sets[mutation_point2])
    solution[mutation_point1] = new_value_1
    solution[mutation_point2] = new_value_2
    return solution

def crossover3(moead, a, b):
    r = random.random()
    cross_point = random.randint(0, len(a))
    if r > 0.5:
        a[: cross_point], b[: cross_point] = b[: cross_point], a[:cross_point]
    else:
        a[cross_point:], b[cross_point:] = b[cross_point:], a[cross_point:]
    return a, b

def cross_mutation(moead, p1, p2, history_solutions):
    y1 = copy.deepcopy(p1)
    y2 = copy.deepcopy(p2)
    c_rate = 1
    m_rate = 0.8
    if np.random.rand() < c_rate:
        # y1, y2 = crossover3(moead, y1, y2)
        y1, y2 = crossover3(moead, y1, y2)
        add_history_solution(y1, history_solutions)
        add_history_solution(y2, history_solutions)
    if np.random.rand() < m_rate:
        y1 = mutate3(moead, y1)
        y2 = mutate3(moead, y2)
        add_history_solution(y1, history_solutions)
        add_history_solution(y2, history_solutions)
    return y1, y2

def add_history_solution(solution,history_solutions):
    if solution not in history_solutions:
        history_solutions.append(solution)


def adrs_update(moead, config, parato_frontier, pareto_frontier_exhaustive, adrs_evolution):
    latency, area = moead.Test_fun.latency(config, moead.entire_ds), moead.Test_fun.area(config, moead.entire_ds)
    ds_point = DSpoint(latency, area, config)
    parato_frontier.append(ds_point)
    parato_frontier, parato_frontier_idx = ADRSunit.pareto_frontier2d(parato_frontier)
    adrs = ADRSunit.adrs2d(pareto_frontier_exhaustive, parato_frontier)
    adrs_evolution.append(adrs)



def select_better(moead, s1, s2, L_constrain, history_solutions):
    N_pare = 5
    while N_pare > 0:
        config1, config2 = cross_mutation(moead, s1, s2, history_solutions)
        if area_1(config1, moead.entire_ds) < area_1(s1, moead.entire_ds) and latency_1(config1, moead.entire_ds) < L_constrain:
            s1 = config1
            N_pare = 5
        if area_1(config2, moead.entire_ds) < area_1(s2, moead.entire_ds) and latency_1(config2, moead.entire_ds) < L_constrain:
            s2 = config2
            N_pare = 5
        N_pare -= 1

    return s1, s2

def global_best(moead, history_solutions, L_constrain):
    # 先将综合过的点按照latency从大到小排序，返回小于L_constrain 的元素。
    sorted_solutions = sorted(history_solutions, key=lambda x: latency_1(x, moead.entire_ds), reverse=True)
    sorted_solutions_1 = [x for x in sorted_solutions if latency_1(x, moead.entire_ds) < L_constrain]
    if not sorted_solutions_1:
        return None
    min_solutions = min(sorted_solutions_1, key=lambda x: area_1(x, moead.entire_ds))
    return min_solutions

def nsga2select(moead, history_solutions, len_pop):
    function1_values = [area_1(history_solutions[i], moead.entire_ds) for i in range(0, len(history_solutions))]
    function2_values = [latency_1(history_solutions[i], moead.entire_ds) for i in range(0, len(history_solutions))]
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values[:], function2_values[:])
    crowding_distance_values2 = []
    new_pop = copy.deepcopy(history_solutions)
    for i in range(0, len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(
            crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution2[i][:]))
    new_solution = []
    for i in range(0, len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [
            index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
            range(0, len(non_dominated_sorted_solution2[i]))]
        # 根据拥挤度对解进行排序
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        # 排序后的帕累托前沿
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                 range(0, len(non_dominated_sorted_solution2[i]))]
        # 反转一下 拥挤度大的解排在前面
        front.reverse()
        for value in front:
            new_solution.append(value)
            if (len(new_solution) == len_pop):
                break
        if (len(new_solution) == len_pop):
            break

    new_pop = [history_solutions[i] for i in new_solution]
    return new_pop

def fast_non_dominated_sort(values1, values2):          # 返回帕累托分级 每个分级有不同的帕累托解
    # 创建一个列表S，其中包含len(values1)个空列表。S[i]用于存储解i支配的解的索引。
    S=[[] for i in range(0,len(values1))]
    # front[i]用于存储在第i层的解的索引。
    front = [[]]

    # n[i] ---- i 被多少解支配
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]      # 等级

    for p in range(0,len(values1)):         # 遍历每个解
        # 初始化s、n
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):    # 遍历values1中的每个元素
            if (values1[p] < values1[q] and values2[p] < values2[q]) or (values1[p] <= values1[q] and values2[p] < values2[q]) or (values1[p] < values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or (values1[q] <= values1[p] and values2[q] < values2[p]) or (values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]     # 创建一个列表包含len（front）个0
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    # 第一个和最后一个的拥挤度都设定为无穷大
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444

    # 遍历每个个体，除第一个和最后一个元素
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

def adrs_culc(moead, solutions, pareto_frontier_exhaustive):
    synthesis_point = []
    for i in solutions:
        synthesis_point.append(DSpoint(moead.Test_fun.latency(i, moead.entire_ds), moead.Test_fun.area(i, moead.entire_ds), i))
    pareto_frontier, pareto_frontier_idx = ADRSunit.pareto_frontier2d(synthesis_point)
    adrs = ADRSunit.adrs2d(pareto_frontier_exhaustive, pareto_frontier)

    return adrs

def envolution(moead, pop):
    L_constrain = float('inf')
    EP_Solution = []
    synthesis_result = moead.synthesis_result
    configurations = moead.configurations
    L_min = min(x[0] for x in synthesis_result)
    entire_ds = moead.entire_ds
    history_solutions = copy.deepcopy(pop)      # 记录所有综合过的点
    N_stag = 0      # 第100次找不到小于latency的点，就停止迭代

    # adrs 模块
    unique_configurations = pd.Series(pop).drop_duplicates().tolist()
    unique_point = []
    for i in unique_configurations:
        unique_point.append(
            DSpoint(moead.Test_fun.latency(i, moead.entire_ds), moead.Test_fun.area(i, moead.entire_ds), i))
    pareto_frontier, pareto_frontier_idx = ADRSunit.pareto_frontier2d(unique_point)
    pareto_frontier_exhaustive, pareto_frontier_exhaustive_idx = ADRSunit.pareto_frontier2d(moead.entire_ds)
    adrs = ADRSunit.adrs2d(pareto_frontier_exhaustive, unique_point)
    # 统计ADRS进化过程
    adrs_evolution = []
    adrs_evolution.append(adrs)



    while L_constrain > L_min:
        N_exit = 10
        best_solution = global_best(moead, history_solutions, L_constrain)

        while N_exit > 0:
            # 随机从列表中选择两个索引项
            s1_idx, s2_idx = random.sample(range(len(pop)), 2)
            # 通过不断交叉变异得到更好的后代
            c1, c2 = select_better(moead, pop[s1_idx], pop[s2_idx], L_constrain, history_solutions)
            pop[s1_idx] = c1
            pop[s2_idx] = c2

            temp_best_solution = global_best(moead, history_solutions, L_constrain)

            if best_solution is None or temp_best_solution is None:
                N_stag += 1
                N_exit -= 1
                continue
            else:
                N_stag = 0

            if(area_1(temp_best_solution, moead.entire_ds) < area_1(best_solution, moead.entire_ds)):
                best_solution = temp_best_solution
                N_exit = 10
            else:
                N_exit -= 1

        if N_stag == 0:
            EP_Solution.append(best_solution)
            L_constrain = latency_1(best_solution,entire_ds)

        if N_stag == 100:
            break
        # pop = nsga2select(moead, history_solutions, len(pop))

        print(L_constrain)
        # print(EP_Solution)
        print("n_of_synthesis:", len(history_solutions))

    for i in range(len(pop), len(history_solutions)):
        solutions = history_solutions[:i+1]
        adrs = adrs_culc(moead, solutions, pareto_frontier_exhaustive)
        adrs_evolution.append(adrs)
    # print("adrs_evolution:", adrs_evolution)

    return EP_Solution, adrs_evolution, history_solutions, len(pop)













