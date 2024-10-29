########################################################################################################################
# Author: Lorenzo Ferretti
# Year: 2018
#
# This files contains some function, such metrics, support functions, etc. useful across different projects.
########################################################################################################################
# from itertools import zip_longest, map
import math
import numpy as np


def pareto_frontier2d(points):
    """
    Function: pareto_frontier2d
    The function given in input a set of points return the 2 dimensional Pareto frontier elements and the corresponding
    indexes respect to the original set of point

    Input: A list of points.
    A list of object characterised by two attributes-area and latency.

    Output: A tuple composed of 2 lists.
    The first list contains the Pareto dominant objects. The second list contains the indexes of the pareto dominant
    objects, respect to the original set of points given in input
    """
    if len(points) == 1:
        return [points[0]], [0]
    indexes = sorted(range(len(points)), key=lambda k: points[k].latency)
    p_idx = []
    p_front = []
    pivot_idx = indexes[0]
    for i in range(1, len(indexes)):
        # If are equal, I search for the minimum value until they become different
        data_pivot = points[pivot_idx]
        d = points[indexes[i]]
        if data_pivot.latency == d.latency:
            if d.area < data_pivot.area:
                pivot_idx = indexes[i]
        else:
            if d.area < data_pivot.area:
                p_idx.append(pivot_idx)
                p_front.append(data_pivot)
                pivot_idx = indexes[i]
        if i == len(indexes)-1 and (len(p_idx) == 0 or pivot_idx != p_idx[-1]):
            p_idx.append(pivot_idx)
            p_front.append(points[pivot_idx])

    return p_front, p_idx


def adrs2d(reference_set, approximate_set):
    """
    Function: adrs2d
    The function given in input a set of reference points and a different set of points calculates the Average Distance
    from Reference Set among the reference set and the approximate one.
    ADRS(Pr, Pa) = 1/|Pa| * sum_Pa( min_Pp( delta(Pr,Pa) ) )
    delta(Pr, Pa) = max(0, ( A(Pa) - A(Pr) ) / A(Pa), ( L(Pa) - L(Pr) ) / L(Pa) )

    Input: 2 list of points.
    A list points representing the reference set and a list of points representing the approximate one.

    Output: ADRS value.
    A value representing the ADRS distance among the two sets, the distance of the approximate one with respect to the
    reference set.
    """
    n_ref_set = len(reference_set)
    n_app_set = len(approximate_set)
    min_dist_sum = 0
    for i in range(0, n_ref_set):
        distances = []
        for j in range(0, n_app_set):
            distances.append(_p2p_distance_2d(reference_set[i], approximate_set[j]))
        min_dist = min(distances)
        min_dist_sum = min_dist_sum + min_dist

    avg_distance = min_dist_sum / n_ref_set
    return avg_distance


def _p2p_distance_2d(ref_pt, app_pt):
    """
    Function: _p2p_distance_2d
    Support function used in ADRS
    Point to point distance for a 2 dimensional ADRS calculation. Implements the delta function of ADRS

    Input: 2 points.
    The reference point and the approximate point. Both are objects characterised by area and latency attributes.

    Output: A float value
    The maximum distance among the 2 dimensions considered (in our case area and latency).
    """
    x = (float(app_pt.latency) - float(ref_pt.latency)) / float(ref_pt.latency)
    y = (float(app_pt.area) - float(ref_pt.area)) / float(ref_pt.area)
    to_find_max = [0, x, y]
    d = max(to_find_max)
    return d

def _mean(l):
    return sum(l) / len(l)

def avg(x):
    x = list(x)
    for i in range(0,len(x)):
        if x[i] is None:
            x[i] = 0
    x = [i for i in x if i is not None]
    return sum(x, 0.0) / len(x)

def zip_longest_fill_last(*args):
    # 找出最长的迭代器的长度
    max_length = max(len(lst) for lst in args)
    # 对每个迭代器进行处理
    for i in range(max_length):
        result = []
        for lst in args:
            # 如果当前迭代器的长度大于i，就取它的第i个元素
            # 否则，取它的最后一个元素
            result.append(lst[i] if i < len(lst) else lst[-1])
        yield list(result)


