
""" crossover and mutation operators """

import math, random, copy

import numpy as np

def single_mutation(variable, mutation_rate,feature_sets,action,mu_value):
    range = feature_sets[action]
    random_value = random.uniform(0, 1)
    a = random.choice(range)
    if random.random() < mutation_rate:
        if random_value > 0.8:
            variable[action] = a
        else:
            variable[action] = mu_value

    return variable

def mutation(variable, mutation_rate,feature_sets,action):
    random_index = action
    a = random.choice(feature_sets[random_index])
    if random.random() < mutation_rate:

        variable[random_index] = a


    return variable


def single_point_crossover(parent1, parent2,action):
    point = action
    return parent1[:point] + parent2[point:]





