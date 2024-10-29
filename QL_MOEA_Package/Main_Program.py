

import os,sys
import copy
import csv
import json
import datasets
import random
from Synthesis import FakeSynthesis
from DS_point import DSpoint
import Utils
from matplotlib import pyplot as plt
from itertools import zip_longest
import time

dirname, filename = os.path.split(os.path.abspath(__file__))
package_path = dirname+'\\QL_MOEA_Package'
sys.path.append(package_path)

from QL_MOEA import Chromosome
from QL_MOEA import AMOEA_MAP_framework
from MOO_functions_ import *

start_time = time.time()  # 记录开始时间
### G E N E R A L    S E T T I N G S ###########################################
arg = {
        "Population size" : 70,
        "Genenration Max" : 300,
    }


benchmark = "get_oracle_activations1_backprop_backprop"
filename1 = "backprop"

    # Extract data from the database
database = datasets.Datasets(benchmark)
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
    entire_ds.append(DSpoint(synthesis_result[i][0], synthesis_result[i][1],  synthesis_result[i][2], list(configurations[i])))

adrs_run_history = []
adrs_time_history = []

with open(f'C:/Users/19519/Desktop/samples/{benchmark}_samples_{arg["Population size"]}.json', 'r') as file:
    init_samples = json.load(file)

for i in range(0, 10):
    Benchmark = MOO()
    Benchmark.Bench_descret_matrix = {}
    P = []

    # framework call
    Hybrid_Optimization = AMOEA_MAP_framework(arg)

    # ==============================================================================
    # Population initilization
    def Automatic_Initialization(arg, i):
        pop_size = arg["Population size"]
        initial_population = []
        samples = init_samples[i]
        for j in range(0, pop_size):
            ind = samples[j]
            initial_population.append(configurations[ind])
        for k in range(pop_size):
            P.append(Chromosome(arg,initial_population[k]))
        return P


    # 初始化
    P = Automatic_Initialization(arg, i)
    # AMOEA-MAP framewor startup
    adrs_evolution, time_history = Hybrid_Optimization.start(P, arg, Benchmark, entire_ds, feature_sets, configurations)
    adrs_run_history.append(adrs_evolution)
    adrs_time_history.append(time_history)

end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time  # 计算运行时间
# with open(f'C:/Users/19519/Desktop/results/QL/{filename1}/run_time_QLearning_{benchmark}.json', 'w') as f:
#     json.dump(elapsed_time, f)
#
# with open(f'C:/Users/19519/Desktop/results/QL/{filename1}/QLearning_{benchmark}.json', 'w') as f:
#     json.dump(adrs_run_history, f)
#
# with open(f'C:/Users/19519/Desktop/results/QL/{filename1}/time_QLearning_{benchmark}.json', 'w') as f:
#     json.dump(adrs_time_history, f)

averages_adrs = list(list(map(Utils.avg, Utils.zip_longest_fill_last(*adrs_run_history))))
plt.title("ADRS evolution")
plt.ylabel("mean ADRS")
plt.xlabel("# of synthesis")
plt.plot(range(arg["Population size"], len(averages_adrs) + arg["Population size"]), averages_adrs)
plt.grid()
# plt.show()
plt.savefig('ADRS evolution.png')

# P_out.sort(key=lambda x: x.fitness[0])
#
# # Save optimal Pareto front
# csv_file = open(dirname+'\\AMOEA_MAP_Pareto_Fronts.csv', 'w')
# for i in range(len(P_out)):
#     for k in range(arg["Number of variables"]):
#         csv_file.write(" " + str(P_out[i].variables[k]) + ", ")
#     for j in range(arg["Number of objectives"]):
#         csv_file.write(" " + str(P_out[i].fitness.values[j]) + ", ")
#     csv_file.write(" " + str(P_out[i].constraint) + "\n")
# csv_file.close()



def find_x_for_y(y_target, averages_adrs, init_num):
    diffs = [abs(y - y_target) for y in averages_adrs]
    min_diff_index = diffs.index(min(diffs))
    return min_diff_index + init_num

x_value1 = find_x_for_y(0.04, averages_adrs, arg["Population size"])
print(f"QLearning x value for y = 0.04 in averages_adrs1: {x_value1}")
print(averages_adrs[-1])
