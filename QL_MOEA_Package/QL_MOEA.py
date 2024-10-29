

import sys, random, math, copy
import json
import numpy as np
from MOO_functions_ import *

import Utils
from operators import *
from DS_point import DSpoint
from matplotlib import pyplot as plt
from itertools import zip_longest
from collections import Counter
from Synthesis import FakeSynthesis

class Chromosome():
    def __init__(self, arg,config):
        self.arg = arg
        self.evaluated = None
        self.variables = config.copy()


    def evaluation(self, Benchmark,entire_ds):
        string_ind = str(self.variables)
        if string_ind in Benchmark.Bench_descret_matrix.keys() :
            self.evaluated = False
        else:
            Benchmark.Bench_descret_matrix[string_ind] = self
            self.evaluated = True



# Q-learning参数
alpha = 0.8  # 学习率
gamma = 0.5  # 折扣因子




class AMOEA_MAP_framework:
    def __init__(self, arg):
        self.Gen_Max = arg["Genenration Max"]
        random.seed()

    def start(self, P, arg, Benchmark,entire_ds,feature_sets, configurations):
        num_feature = len(feature_sets)
        synthesized_pop = []
        pareto_frontier = []
        temp_adrs = 99999999
        Q_table = np.zeros((3, 3))

        num_of_iterations = 1

        all_time = 0
        number_of_synthesis = 0

        pareto_frontier_exhaustive,pareto_frontier_exhaustive_idx = Utils.pareto_frontier2d(entire_ds)

        hls = FakeSynthesis(entire_ds)
        # 记录需要被综合的配置
        # 评估，得到面积延迟
        sampled_configurations_synthesised = []
        for s in P:
            c = s.variables
            latency, area, execute_time = hls.synthesise_configuration(c)
            synthesised_configuration = DSpoint(latency, area, execute_time, c)
            sampled_configurations_synthesised.append(synthesised_configuration)
            synthesized_pop.append(synthesised_configuration)
            s.evaluation(Benchmark, entire_ds)
            if s.evaluated:
                number_of_synthesis += 1
                all_time += execute_time

        # 评估，对初始种群计算帕累托前沿

        pareto_frontier, pareto_frontier_idx = Utils.pareto_frontier2d(sampled_configurations_synthesised)
        adrs_evolution = []
        time_evolution = []
        adrs = Utils.adrs2d(pareto_frontier_exhaustive, pareto_frontier)
        # ADRS after initial sampling
        adrs_evolution.append(adrs)
        time_evolution.append(all_time)

        del P[:]
        for p in pareto_frontier:
            v = p.configuration
            c = Chromosome(arg, v)
            P.append(c)


        ## Generations
        state = 0
        action = 0

        for i in range(self.Gen_Max):
            current_adrs = adrs

            print("迭代次数第" + str(num_of_iterations) + "次")
            num_of_iterations += 1
            # 选择
            selected_parents = P
            selected_parents_var = [copy.deepcopy(i.variables) for i in selected_parents]
            # 交叉

            for parent1 in selected_parents_var:
                parent2 = random.choice(selected_parents_var)
                position = random.randint(0, len(feature_sets) - 1)
                child = single_point_crossover(parent1, parent2, position)
                child_c = Chromosome(arg, child)


                child_c.evaluation(Benchmark, entire_ds)
                if child_c.evaluated:
                    number_of_synthesis += 1
                    print("交叉产生的新孩子个数" + str(number_of_synthesis))
                    latency, area, execute_time = hls.synthesise_configuration(child)
                    synthesised_configuration = DSpoint(latency, area, execute_time, child)
                    pareto_frontier.append(synthesised_configuration)
                    synthesized_pop.append(synthesised_configuration)
                    pareto_frontier, pareto_frontier_idx = Utils.pareto_frontier2d(
                        pareto_frontier)

                    adrs = Utils.adrs2d(pareto_frontier_exhaustive, pareto_frontier)
                    all_time += execute_time

                    if adrs < temp_adrs:
                        temp_adrs = adrs
                    else:
                        adrs = temp_adrs

                    print(adrs)
                    adrs_evolution.append(adrs)
                    time_evolution.append(all_time)


            # 变异
            for cr in selected_parents_var:
                if(action == 0):
                    position = random.randint(0, len(feature_sets) - 1)
                    child = mutation(cr, 0.8, feature_sets, position)
                    child_mu = Chromosome(arg, child)

                if(action == 1):
                    list1 = []
                    for s in selected_parents_var:
                        list1.append(s)
                    list2 = find_column_modes_with_counts(list1)
                    list2 = find_indices_excluding_two_max(list2)
                    position = random.choice(list2)
                    child = mutation(cr, 0.8, feature_sets, position)
                    child_mu = Chromosome(arg, child)

                if (action == 2):
                    list1 = []
                    for s in selected_parents_var:
                        list1.append(s)
                    list2 = find_column_modes_with_counts(list1)
                    list2 = find_indices_excluding_one_max(list2)
                    position = random.choice(list2)
                    child = mutation(cr, 0.8, feature_sets, position)
                    child_mu = Chromosome(arg, child)



                child_mu.evaluation(Benchmark, entire_ds)
                if child_mu.evaluated:
                    number_of_synthesis += 1
                    print("变异产生的新孩子个数" + str(number_of_synthesis))
                    latency, area, execute_time = hls.synthesise_configuration(child)
                    synthesised_configuration = DSpoint(latency, area, execute_time, child)
                    synthesized_pop.append(synthesised_configuration)
                    pareto_frontier.append(synthesised_configuration)
                    pareto_frontier, pareto_frontier_idx = Utils.pareto_frontier2d(
                        pareto_frontier)

                    adrs = Utils.adrs2d(pareto_frontier_exhaustive, pareto_frontier)
                    all_time += execute_time

                    if adrs < temp_adrs:
                        temp_adrs = adrs
                    else:
                        adrs = temp_adrs
                    print(adrs)
                    adrs_evolution.append(adrs)
                    time_evolution.append(all_time)





            del P[:]
            P1 = []
            for p in pareto_frontier:
                c = Chromosome(arg, p.configuration)
                P1.append(c)
            P = P1


            if adrs < current_adrs:
                next_state = 1
                update_q_table(state,action,10,next_state,Q_table)
                state = 1
                action = choose_mu_action(state,Q_table)

            else:
                next_state = 2
                update_q_table(state,action,-10,next_state,Q_table)
                state = 2
                action = choose_mu_action(state,Q_table)




        return adrs_evolution, time_evolution






def choose_mu_action(state,Q_table):

    return np.argmax(Q_table[state])

def update_q_table(state,action,reward,next_state,Q_table):
    # 找到具有最高 Q 值的动作
    best_next_action = np.argmax(Q_table[next_state])
    new_qValues = (1 - alpha) * Q_table[(state, action)] + alpha * (
                reward + gamma * Q_table[(next_state, best_next_action)])
    # if reward == 0:
    #     Q_table[(state, action)] -= 100
    # else:
    Q_table[(state, action)] = new_qValues


def find_column_modes_with_counts(A):
    # Initialize an empty list B to store the count of the mode for each column
    B = []

    # Transpose A to get columns easily (zip(*A) helps with that)
    for col in zip(*A):
        # Use Counter to find the most common element in the column
        counter = Counter(col)
        # Get the most common element (mode) and its count
        mode, count = counter.most_common(1)[0]
        # Append the count of the mode to B
        B.append(count)

    return B

def find_indices_excluding_two_max(B):
    # Sort the indices of B based on the values in B
    sorted_indices = sorted(range(len(B)), key=lambda i: B[i])
    # Exclude the last two indices which correspond to the two largest elements
    return sorted_indices[:-2]

def find_indices_excluding_one_max(B):
    # Sort the indices of B based on the values in B
    sorted_indices = sorted(range(len(B)), key=lambda i: B[i])
    # Exclude the last two indices which correspond to the two largest elements
    return sorted_indices[:-1]