# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan
# Supervisor: Prof. Manoj Kumar Tiwari

# 还存在问题： 会重复选择点

#Importing required modules
import math
import random
import matplotlib.pyplot as plt
import datasets
from sythesis import FakeSynthesis
from DSpoint import DSpoint
import ADRSunit
import numpy as np

def call_data(dataname):
    # 引入指令、configuration、area、latency
    # benchmark = ["ChenIDCt", "adpcm_decode", "adpcm_encode", "Autocorrelation", "Reflection_coefficients"]
    benchmark = [dataname]
    for b in benchmark:
        database = datasets.Datasets(b)
        synthesis_result = database.benchmark_synthesis_results
        configurations = database.benchmark_configurations
        feature_sets = database.benchmark_feature_sets
        directives = database.benchmark_directives
        entire_ds = []
        for i in range(0, len(synthesis_result)):
            # 存储所有可能的综合点信息
            # [i][0]储存latency，[i][1]储存area，用DSpoint存储信息
            entire_ds.append(DSpoint(synthesis_result[i][0], synthesis_result[i][1], list(configurations[i])))

    return configurations, feature_sets, entire_ds
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

    for p in range(0,len(values1)):         # 遍历values1中的每个元素
        # 初始化s、n
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):    # 遍历values1中的每个元素
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
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
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

#Function to carry out the crossover
def crossover(a,b,feature_sets):
    r=random.random()
    cross_point = random.randint(0, len(a))
    if r>0.6:
        return mutation(a[:cross_point] + b[cross_point:],feature_sets)
    else:
        return mutation(b[:cross_point] + a[cross_point:],feature_sets)

#Function to carry out the mutation operator
def mutation(solution,feature_sets):
    # 随机生成变异点
    mutation_point = random.randint(0, len(solution) - 1)
    new_value = random.choice(feature_sets[mutation_point])
    solution[mutation_point] = new_value
    return solution

#Main program starts here
pop_size = 100
max_gen =  2000



#Initialization
configurations, feature_set, entire_ds = call_data("ChenIDCt")
# 生成初始解
solution=[random.choice(configurations) for i in range(0,pop_size)]
gen_no=0
while(gen_no<max_gen):
    # 计算种群中每个个体的目标函数值
    function1_values = [area(solution[i],entire_ds)for i in range(0,pop_size)]
    function2_values = [latency(solution[i],entire_ds) for i in range(0, pop_size)]
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])    # 非支配排序，得到各个帕累托前沿
    #print("The best front for Generation number ",gen_no, " is")
    # for valuez in non_dominated_sorted_solution[0]:
    #     print(solution[valuez],end=" ")
    # 输出最优帕累托前沿
    #print("\n")
    crowding_distance_values=[]     # 初始化拥挤度列表

    # 计算每个帕累托前沿的拥挤度。
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
    solution2 = solution[:]     # 复制当前种群，准备生成新的种群。
    #Generating offsprings
    while(len(solution2)!=2*pop_size):      # 通过随机选择和交叉操作，生成新的个体，直到新种群的大小达到原种群的两倍。（精英保留策略）
        a1 = random.randint(0,pop_size-1)
        b1 = random.randint(0,pop_size-1)
        solution2.append(crossover(solution[a1],solution[b1],feature_set))
    # 计算新个体的目标函数值
    function1_values2 = [area(solution2[i],entire_ds)for i in range(0,2*pop_size)]
    function2_values2 = [latency(solution2[i],entire_ds) for i in range(0, 2 * pop_size)]
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

#Lets plot the final front now
function1 = [i * -1 for i in function1_values]
function2 = [j * -1 for j in function2_values]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)
plt.show()

# 计算ADRS
_, _, entire_ds = call_data("ChenIDCt")
pareto_frontier_exhaustive, pareto_frontier_exhaustive_idx = ADRSunit.pareto_frontier2d(entire_ds)
pareto_frontier = []
# 得到帕累托边界列表
solution = np.unique(solution)
for i in solution:
    area1 = area(i,entire_ds)
    latency1 = latency(i,entire_ds)
    ds_point = DSpoint(latency1, area1, i)
    pareto_frontier.append(ds_point)

pareto_frontier,pareto_frontier_idx = ADRSunit.pareto_frontier2d(pareto_frontier)
adrs = ADRSunit.adrs2d(pareto_frontier_exhaustive, pareto_frontier)
print(adrs)
