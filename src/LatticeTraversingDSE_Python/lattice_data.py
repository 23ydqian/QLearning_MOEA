########################################################################################################################
# Author: Lorenzo Ferretti
# Year: 2018
#
# This files contains the Lattice class. The Lattice class describes the design space and contains the information
# related to the explored configurations.
########################################################################################################################

import lattice_tree
import math
import itertools
import numpy


class Lattice:

    def __init__(self, lattice_descriptor, max_distance):
        self.original_descriptor = lattice_descriptor
        self.discretized_descriptor = self.discretize_dataset(lattice_descriptor)
        self.lattice = lattice_tree.Tree('lattice')
        # self.radii_struct = self.radii_vectors()
        self.max_distance = max_distance

    def discretize_dataset(self, lattice_descriptor):
        discretized_feature = []
        for feature in lattice_descriptor:
            # tmp = []
            # for x in self._frange(0, 1, 1. / (len(feature) - 1)):
            #     tmp.append(x)
            tmp = numpy.linspace(0, 1, len(feature))
            discretized_feature.append(tmp)

        return discretized_feature

    def _frange(self, start, stop, step):
        x = start
        output = []
        while x <= stop:
            output.append(x)
            x += step
        output[-1] = 1
        return output

    # 不理解
    # 将其还原成原来的配置
    def revert_discretized_config(self, config):
        tmp = []
        # 遍历每一种指令
        for i in range(0, len(config)):
            # 遍历每种指令的取值
            for j in range(0, len(self.discretized_descriptor[i])):
                # discretized_descriptor[i][j] 表示 第i各指令里面的第j个取值（经离散化）  numpy.isclose(a, b, atol = c) 指的是a,b的值是否接近，容差为c
                if numpy.isclose(self.discretized_descriptor[i][j], config[i], atol=0.000001):
                    # 原始取值与离散取值建立映射
                    tmp.append(self.original_descriptor[i][j])
                    break
        return tmp

    def revert_original_config(self, config):
        tmp = []
        for i in range(0, len(config)):
            for j in range(0, len(self.original_descriptor[i])):
                if self.original_descriptor[i][j] == config[i]:
                    tmp.append(self.discretized_descriptor[i][j])
                    break
        return tmp

    def beta_sampling(self, a, b, n_sample):
        samples = []
        for i in range(0, n_sample):
            s = []      # 存储样本
            search = True   # 无限循环
            while search:
                for d_set in self.discretized_descriptor:
                    r = numpy.random.beta(a, b, 1)[0]    # beta分布， a, b 是beta分布的两个参数， 1是值取一个值， [0] 是指获取生成数数组的第一个元素
                    s.append(self._find_nearest(d_set, r))      # 从d_set中找到与r最接近的元素添加到s之中

                if s in samples:
                    s = []
                    continue
                else:
                    # samples数组中没有s，则添加
                    samples.append(s)
                    break
        return samples

    # def beta_sampling_from_probability(self, sampled_probability):
    #     samples = []
    #     for sp in sampled_probability:
    #         sp.pop(0)
    #         sp.pop(0)
    #         s = []
    #         search = True
    #         while search:
    #             for d in xrange(len(self.discretized_descriptor)):
    #                 d_set = self.discretized_descriptor[d]
    #                 r = sp[d]
    #                 d_set_array = numpy.array(d_set)
    #                 idx = (numpy.abs(d_set_array - r)).argmin()
    #                 s.append(d_set[idx])
    #
    #             samples.append(s)
    #             break
    #     return samples


    def _find_nearest(self, array, value):
        # array中找到最接近value的那个元素
        idx = (numpy.abs(array-value)).argmin()  # array-value，找到最小的那个元素所在的索引idx，也就是最接近value的值
        return array[idx]
