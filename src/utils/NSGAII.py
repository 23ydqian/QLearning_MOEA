import math
import random
import matplotlib.pyplot as plt
import datasets
from src.HLS.sythesis import FakeSynthesis
from src.HLS.DSpoint import DSpoint
import src.HLS.ADRSunit as ADRSunit
import numpy as np
import pandas as pd
import copy


#First function to optimize
def area(config,entire_ds):
    # 引入指令、configuration、area、latency
    # benchmark = ["ChenIDCt", "adpcm_decode", "adpcm_encode", "Autocorrelation", "Reflection_coefficients"]
    hls = FakeSynthesis(entire_ds)
    area = hls.synthesise_configuration(config)[1]
    return area


#Second function to optimize
def latency(config,entire_ds):
    # 引入指令、configuration、area、latency
    # benchmark = ["ChenIDCt", "adpcm_decode", "adpcm_encode", "Autocorrelation", "Reflection_coefficients"]
    hls = FakeSynthesis(entire_ds)
    latency = hls.synthesise_configuration(config)[0]
    return latency

#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values （value：list）， 返回一个排好序的列表
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
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

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]     # 创建一个列表包含len（front）个0
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    # 第一个和最后一个的拥挤度都设定为无穷大
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444

    # 遍历每个个体，除第一个和最后一个元素
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1)+0.00001)
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2)+0.00001)
    return distance

#Function to carry out the crossover
def crossover3(moead, a, b):
    r = random.random()
    cross_point = random.randint(0, len(a))
    if r > 0.5:
        a[: cross_point], b[: cross_point] = b[: cross_point], a[:cross_point]
    else:
        a[cross_point:], b[cross_point:] = b[cross_point:], a[cross_point:]
    return a, b

def mutate3(moead, solution):
    feature_sets = moead.feature_sets
    mutation_point1, mutation_point2 = random.sample(range(0, len(solution) - 1), 2)
    new_value_1 = random.choice(feature_sets[mutation_point1])
    new_value_2 = random.choice(feature_sets[mutation_point2])
    solution[mutation_point1] = new_value_1
    solution[mutation_point2] = new_value_2
    return solution

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
#Main program starts here
# pop_size = 30
max_gen =  20

def adrs_culc(moead, solutions, pareto_frontier_exhaustive):
    synthesis_point = []
    for i in solutions:
        synthesis_point.append(DSpoint(moead.Test_fun.latency(i, moead.entire_ds), moead.Test_fun.area(i, moead.entire_ds), i))
    pareto_frontier, pareto_frontier_idx = ADRSunit.pareto_frontier2d(synthesis_point)
    adrs = ADRSunit.adrs2d(pareto_frontier_exhaustive, pareto_frontier)

    return adrs


def evolution(moead, pop):
    configurations = moead.configurations
    feature_set = moead.feature_sets
    entire_ds = moead.entire_ds
    # 生成初始解
    solution = pop
    gen_no=0
    pop_size = len(solution)

    # 统计adrs值
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

    history_solutions = copy.deepcopy(pop)  # 记录所有综合过的点


    while(gen_no<max_gen):
        # 计算种群中每个个体的目标函数值
        function1_values = [area(solution[i],entire_ds)for i in range(0,pop_size)]
        function2_values = [latency(solution[i],entire_ds) for i in range(0, pop_size)]
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])    # 非支配排序，得到各个帕累托前沿

        crowding_distance_values=[]     # 初始化拥挤度列表

        # 计算每个帕累托前沿的拥挤度。
        for i in range(0,len(non_dominated_sorted_solution)):
            crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
        solution2 = solution[:]     # 复制当前种群，准备生成新的种群。
        #Generating offsprings 这里有问题
        while(len(solution2)<=2*pop_size):      # 通过随机选择和交叉操作，生成新的个体，直到新种群的大小达到原种群的两倍。（精英保留策略）
            a1 = random.randint(0,pop_size-1)
            b1 = random.randint(0,pop_size-1)
            config1, config2 = cross_mutation(moead, solution[a1], solution[b1], history_solutions)
            solution2.append(config1)
            solution2.append(config2)
        # 计算新个体的目标函数值
        function1_values2 = [area(solution2[i],entire_ds)for i in range(0,len(solution2))]
        function2_values2 = [latency(solution2[i],entire_ds) for i in range(0, len(solution2))]
        # 重新进行非支配排序 和拥挤度的计算
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
        crowding_distance_values2=[]
        for i in range(0,len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
        new_solution= []
        for i in range(0,len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
            # 根据拥挤度对解进行排序
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            # 排序后的帕累托前沿
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
            # 反转一下 拥挤度大的解排在前面
            front.reverse()
            for value in front:
                new_solution.append(value)
                if(len(new_solution)==pop_size):
                    break
            if (len(new_solution) == pop_size):
                break
        # 更新种群
        solution = [solution2[i] for i in new_solution]
        gen_no = gen_no + 1

    for i in range(len(pop), len(history_solutions)):
        solutions = history_solutions[:i+1]
        adrs = adrs_culc(moead, solutions, pareto_frontier_exhaustive)
        adrs_evolution.append(adrs)

    return adrs_evolution



