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
    # 以latency为准从小到大排序
    indexes = sorted(range(len(points)), key=lambda k: points[k].latency)
    p_idx = []
    p_front = []
    pivot_idx = indexes[0]
    # 遍历索引列表 从第二个开始
    for i in range(1, len(indexes)):
        # If are equal, I search for the minimum value until they become different
        # 记录最小延迟的点
        data_pivot = points[pivot_idx]
        # 得到当前索引i的点
        d = points[indexes[i]]
        if data_pivot.latency == d.latency:
            # 如果延迟的值相等，比较area
            if d.area < data_pivot.area:
                # 如果此前记录的最小延迟点的area更大，更新i点为pivot_idx
                pivot_idx = indexes[i]
        else:
            if d.area < data_pivot.area:
                # 不相等，则将pivot_idx添加到p_index 列表
                p_idx.append(pivot_idx)
                p_front.append(data_pivot)
                # 更新pivot进行下一轮的比较
                pivot_idx = indexes[i]
        # 最后一个索引，且 最后一个帕累托前沿不等于pivot_idx（说明最后一个结点经过比较胜出）   做过修改，原始代码看lattice
        if i == len(indexes) - 1:
            if p_idx:  # 检查 p_idx 是否为空
                if pivot_idx != p_idx[-1]:
                    # 将其添加到 Pareto 前沿列表中
                    p_idx.append(pivot_idx)
                    p_front.append(points[pivot_idx])
            else:
                # 如果 p_idx 为空，直接添加到 Pareto 前沿列表中
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
    # 集合元素个数
    n_ref_set = len(reference_set)
    n_app_set = len(approximate_set)
    min_dist_sum = 0
    # 遍历所有元素
    for i in range(0, n_ref_set):
        # 生成distance列表
        distances = []
        # 遍历每个近似元素
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