########################################################################################################################
# Author: Lorenzo Ferretti
# Year: 2018
#
# This files contains the Sphere Tree class. This class defines the sphere used to perform the local search in the
# neighbourhood of the configuration to explore. A tree similar to the one for the explored configuration is used to
# keep track of the configuration to visit during the local exploration
########################################################################################################################

# 这个就是VCT
from lattice_tree import Node, Tree
import copy
import math
import numpy as np


class SphereTree:
    def __init__(self, configuration, lattice):
        self.root = Node(configuration)
        self.lattice = lattice
        # 这个半径设置有点疑惑
        self.radius = lattice.discretized_descriptor[0][1]
        self.min_radius = lattice.discretized_descriptor[0][1]
        self.min_increment = 0.05
        self.sphere_elements = []
        self.closest_distances, self.closest_elements_idx, self.n_of_children = self.get_closest_sphere_elements()
        if len(self.closest_distances) == 0:
            self.random_closest_element = None
        else:
            self.random_closest_element = self.get_closest_random_element().get_data()

    def get_closest_sphere_elements(self):
        # 标记已访问过的树
        visited_tree = Tree("visited")
        # 开始以root为树根开始探索可用configuration
        self.visit_config(self.root, visited_tree)
        # 当探索球的元素为0时，说明此时半径内所有点已被探索，增加半径，扩大设计空间
        while len(self.sphere_elements) == 0:
            # 增加半径
            self.radius = self.radius + self.min_increment
            if self.radius > self.lattice.max_distance:
                # 超过最大限度半径则结束探索
                break
            # 重新开始探索
            visited_tree = Tree("visited")
            self.visit_config(self.root, visited_tree)
            # print "Number of children: ", visited_tree.get_n_of_children()
            # print "Sphere elements: ", len(self.sphere_elements)
        # 调用sort_sphere_elements() 得到最近的距离以及对应的的索引
        closest_distances, closest_elements_idx = self.sort_sphere_elements()
        return closest_distances, closest_elements_idx, visited_tree.get_n_of_children()

    def visit_config(self, starting_config, visited_tree):
        children = []
        config = starting_config.get_data()
        # 将开始点添加到已访问树
        visited_tree.add_config(config)
        # 遍历config中的每个指令
        for i in range(0, len(config)):
            delta = self.lattice.discretized_descriptor[i][1]       # 当前指令取值的单位长度
            # 加一个步长后的configuration
            cfg_plus = copy.copy(config)
            # 给当前的指令指 加上一个单位长度
            value_plus = cfg_plus[i] + delta
            # 在discretized_descriptor[i] 标准指令值中 找到与 value_plus最近的值放入到cfg_plus[i]中
            cfg_plus[i] = self.lattice._find_nearest(self.lattice.discretized_descriptor[i], np.float64(value_plus))
            # 减一个步长后的configuration
            cfg_minus = copy.copy(config)
            # 给当前的指令指 减上一个单位长度
            value_minus = cfg_minus[i] - delta
            # 在discretized_descriptor[i] 标准指令值中 找到与 value_minus 最近的值放入到cfg_minus[i]中
            cfg_minus[i] = self.lattice._find_nearest(self.lattice.discretized_descriptor[i], np.float64(value_minus))

            # Generate the new config to add
            # cfg_plus 对应当前configuration的当前指令加上其对应步长后的新configuration。
            # 使用_add_config ，避免添加欧式距离大于搜索半径的情况
            config_to_append_plus = self._add_config(cfg_plus)
            # 如果 config_to_append_plus结点添加成功
            if config_to_append_plus is not None:
                # 将其添加到孩子结点（对应单个指令增加步长后的情况）
                children.append(config_to_append_plus)

            config_to_append_minus = self._add_config(cfg_minus)
            # 如果 关于减少一个单位长度的结点添加成功
            if config_to_append_minus is not None:
                # 添加到孩子结点
                children.append(config_to_append_minus)
            # 当存在孩子结点，说明有可用的点
            while len(children) > 0:
                # 弹出第一个孩子结点
                c = children.pop(0)
                # 避免重复visit，先做一个判断当前visited tree中不存在此结点
                if not visited_tree.exists_config(visited_tree, c.get_data()):
                    # 以该孩子结点为开始点继续探索下一层visit
                    self.visit_config(c, visited_tree)
                    # 当前结点不存在探索树中
                    if not self.lattice.lattice.exists_config(self.lattice.lattice, c.get_data()):
                        # 添加到搜索球元素中
                        self.sphere_elements.append(c)

    def _add_config(self, config):
        # 获取distance self与config的距离
        distance = self._get_distance(config)
        if not np.isclose(distance, self.radius) and distance > self.radius:
            # 如果distance不接近半径，并且 distance大于半径，返回none
            return None
        else:
            n = Node(config)
            return n

    def _get_distance(self, config):
        tmp = 0
        for i in range(len(config)):
            # 遍历config中的每个指令i
            # 得到root_config的数据，用于计算与对应指令的距离
            root_config = self.root.get_data()
            # 计算每个指令与跟根节点对应指令的距离求和
            tmp += ((root_config[i]) - (config[i])) ** 2
        # 开根号，计算欧氏距离
        tmp = math.sqrt(tmp)
        return tmp

    def _is_in_sphere(self, config):
        # 遍历搜索球元素
        for e in self.sphere_elements:
            if config == e.get_data():
                return True
        return False

    def sort_sphere_elements(self):
        distances = []
        for i in self.sphere_elements:
            # 遍历搜索球中所有元素，将其距离放入到distances列表中
            c = i.get_data()
            distances.append(self._get_distance(c))

        # 返回一个距离里列表，并根据根据距离大小排序的distance索引列表
        return distances, sorted(range(len(distances)), key=lambda k: distances[k])

    # 由于最小距离对应的指令不止一个，所以可以随机的从当中选择。
    def get_closest_random_element(self):
        tmp = []
        # 得到最小distance
        min_distance = min(self.closest_distances)
        # 以closest_elements_idx（距离大小从小到大对应的distances）
        for i in self.closest_elements_idx:
            if self.closest_distances[i] == min_distance:
                # 如果当前distance 与 最小距离相等
                # 添加到tmp列表
                tmp.append(i)
        # 随机的从tem中取元素
        r = np.random.randint(0, len(tmp))
        return self.sphere_elements[r]

