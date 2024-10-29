import random
import numpy as np

# config: solution, pi: the pi th subproblem, u: updaterate
def EDA_Reproduce(config, pi, u, probabilities,moead):
    feature_sets = moead.feature_sets
    for d in range(len(config)):
        r1 = random.random()
        if(r1 < u):
            # 在第d个维度中运用概率矩阵选出新的解。
            config[d] = generate_solution(probabilities,feature_sets, d)

    return config

# 更新策略1
def probabilities_update(pi, moead):
    # 调取特征集，由列表组成，每个元素也是列表（对应每个指令的取值）
    feature_set = moead.feature_sets
    probabilities = [[] for _ in range(len(feature_set))]
    Bi = moead.W_Bi_T[pi]
    # 收集领域解
    Solution = []
    for i in Bi:
        Xk = moead.Pop[i]
        Solution.append(Xk)
    # 更新概率矩阵
    for i in range(len(feature_set)):    # 遍历每个维度
        # 得到该维度下可取值
        features = feature_set[i]
        # 创建一个字典来存储每个值的出现次数
        counts = {feature: 0 for feature in features}
        # 计算每个值的出现次数
        for solution in Solution:
            counts[solution[i]] += 1
        # 计算并存储每个值的概率
        for feature in features:
            probabilities[i].append(counts[feature] / len(Solution))

    return probabilities

def generate_solution(probabilities, feature_set, i):
    # 从第i个维度的特征集中，根据概率分布选择一个索引
    index = np.random.choice(len(feature_set[i]), p=probabilities[i])
    i_val = feature_set[i][index]
    return i_val



