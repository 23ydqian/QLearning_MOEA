########################################################################################################################
# Author: Lorenzo Ferretti
# Year: 2018
#
# This files contains the class datasets containing the different datasets name, types and function used to retrieve
# the data from the exploration
########################################################################################################################
import sqlite3
import numpy as np


class Datasets:

    def __init__(self, name):
        self.benchmark_name = name
        # benchmark对应的种类，db表示database
        self.benchmarks_dict_type = {"ChenIDCt": "db", "adpcm_decode": "db", "adpcm_encode": "db",
                                     "Reflection_coefficients": "db",
                                     "Autocorrelation": "db", "test": None}

        # 得到benchmark的数据  调用get.data方法
        self.benchmarks_dictionary_data = {"ChenIDCt": self.get_chenidct_data,
                                           "adpcm_decode": self.get_decode_data,
                                           "adpcm_encode": self.get_encode_data,
                                           "Reflection_coefficients": self.get_reflection_data,
                                           "Autocorrelation": self.get_autocorr_data,
                                           "Autocorrelation_extended": None,
                                           "adpcm_decode_ck": None}

        # Definition of adpcm_decode_ck extended_experiments
        # Decode 的benchmark
        # 循环展开因子
        self.adpcm_decode_ck_unrolling = {'mac_loop': [0, 2, 5, 10],  # 4
                                          'update_loop': [0, 2, 5, 10],  # 4
                                          'main_loop': [0, 5, 10, 25, 50]}  # 5
        # 数组的bundling
        self.adpcm_decode_ck_bundling = [("compressed", "result"), ((0, 0), (0, 1))]  # 2
        self.adpcm_decode_ck_bundling_config = {'bundling': [0, 1]}

        # 是否使用内联函数
        self.adpcm_decode_ck_inlining = {'upzero': [0, 1], # 2
                                         'quantl': [0, 1]}  # 2

        #时钟
        self.adpcm_decode_ck_clocks = {'clock': [5, 10, 15, 20]}  # 4

        # Decode的指令分类
        self.adpcm_decode_ck = {'unrolling': self.adpcm_decode_ck_unrolling,
                                        'inlining': self.adpcm_decode_ck_inlining,
                                        # 'pipelining': self.autcorrelation_extended_pipeline,
                                        'bundling': self.adpcm_decode_ck_bundling_config,
                                        # 'partitioning': self.autcorrelation_extended_partitioning,
                                        'clock': self.adpcm_decode_ck_clocks}

        # 调用每个指令里面的细分类 得到所有指令
        self.adpcm_decode_ck_directives_ordered = [
            ('unrolling-main_loop', self.adpcm_decode_ck['unrolling']['main_loop']),
            ('unrolling-update_loop', self.adpcm_decode_ck['unrolling']['update_loop']),
            ('unrolling-mac_loop', self.adpcm_decode_ck['unrolling']['mac_loop']),
            ('clock-clock', self.adpcm_decode_ck['clock']['clock']),
            ('inlining-upzero', self.adpcm_decode_ck['inlining']['upzero']),
            ('inlining-quantl', self.adpcm_decode_ck['inlining']['quantl']),
            ('bundling-sets', self.adpcm_decode_ck['bundling']['bundling']),
        ]

        # ------------------------------------------------------------------------------------
        # Definition of autocorrelation_extended_experiments
        # Autocorr扩展的 benchmark   （应该是用于证明lattice可以应用于超大型设计空间）

        # 关于循环展开因子的指令
        self.autcorrelation_extended_unrolling = {'max_loop': [0, 2, 4, 8, 16, 32, 40, 80, 160],  # 9
                                                  'gsm_mult_loop': [0, 2,  4, 8, 16, 32, 40, 80, 160],  # 9
                                                  'init_zero_loop': [0, 3, 9],  # 3
                                                  'compute_loop': [0, 2, 4, 8, 19, 38, 76, 152],  # 8
                                                  'left_shift_loop': [0, 3, 9],  # 3
                                                  'rescaling_loop': [0, 2, 4, 8, 16, 32, 40, 80, 160]}  # 9

        # 关于内联函数是否使用
        self.autcorrelation_extended_inlining = {'gsm_norm': [0, 1]}  # 2

        # pipeline 不知为什么要注释掉

        # self.autcorrelation_extended_pipeline = {'gsm_norm': [0, 1]  # 2
        #                                          'max_loop': [0, 1],  # 2
        #                                          'gsm_mult_loop': [0, 1],  # 2
        #                                          'init_zero_loop': [0, 1],  # 2
        #                                          'compute_loop': [0, 1],  # 2
        #                                          'left_shift_loop': [0, 1],  # 2
        #                                          'rescaling_loop': [0, 1]}  # 2

        # 需要手动定义 bundling
        # This actually need to be defined manually
        self.autcorrelation_extended_bundling = [("s", "L_ACF"), ((0, 0), (0, 1))]  # 2
        # self.autcorrelation_extended_bundling_sets = [(0, 0), (0, 1)]  # 2

        self.autcorrelation_extended_bundling_config = {'bundling': [0, 1]}

        #partition
        # self.autcorrelation_extended_partitioning = {'s': [0, 2, 4, 8]}  # 5
                                                        #'L_ACF': [0, 3, 9],  # 3

        # 时钟数
        self.autcorrelation_extended_clocks = {'clock': [5, 10, 15, 20, 25, 30, 35, 40]}  # 8

        # 指令种类
        self.autcorrelation_extended = {'unrolling': self.autcorrelation_extended_unrolling,
                                        'inlining': self.autcorrelation_extended_inlining,
                                        # 'pipelining': self.autcorrelation_extended_pipeline,
                                        'bundling': self.autcorrelation_extended_bundling_config,
                                        # 'partitioning': self.autcorrelation_extended_partitioning,
                                        'clock': self.autcorrelation_extended_clocks}

        # 循环展开因子与pipeline的依赖关系
        self.directive_dependences = [('unrolling-max_loop', 'pipelinining-max_loop'),
                                      ('unrolling-gsm_mult_loop', 'pipelinining-gsm_mult_loop'),
                                      ('unrolling-left_shift_loop', 'pipelinining-left_shift_loop'),
                                      ('unrolling-rescaling_loop', 'pipelinining-rescaling_loop'),
                                      ('unrolling-compute_loop', 'pipelinining-compute_loop'),
                                      ('unrolling-init_zero_loop', 'pipelinining-init_zero_loop')]

        # 所有指令
        self.autcorrelation_extended_directives_ordered = [
            ('unrolling-max_loop', self.autcorrelation_extended['unrolling']['max_loop']),
            ('unrolling-rescaling_loop', self.autcorrelation_extended['unrolling']['rescaling_loop']),
            ('unrolling-gsm_mult_loop', self.autcorrelation_extended['unrolling']['gsm_mult_loop']),
            ('clock-clock', self.autcorrelation_extended['clock']['clock']),
            ('unrolling-compute_loop', self.autcorrelation_extended['unrolling']['compute_loop']),
            # ('partitioning-s', self.autcorrelation_extended['partitioning']['s']),
            ('unrolling-init_zero_loop', self.autcorrelation_extended['unrolling']['init_zero_loop']),
            ('unrolling-left_shift_loop', self.autcorrelation_extended['unrolling']['left_shift_loop']),
            # ('pipelining-left_shift_loop', self.autcorrelation_extended['pipelining']['left_shift_loop']),
            # ('pipelining-gsm_mult_loop', self.autcorrelation_extended['pipelining']['gsm_mult_loop']),
            # ('pipelining-gsm_norm', self.autcorrelation_extended['pipelining']['gsm_norm']),
            ('inlining-gsm_norm', self.autcorrelation_extended['inlining']['gsm_norm']),
            ('bundling-sets', self.autcorrelation_extended['bundling']['bundling']),
            # ('pipelining-rescaling_loop', self.autcorrelation_extended['pipelining']['rescaling_loop']),
            # ('pipelining-compute_loop', self.autcorrelation_extended['pipelining']['compute_loop']),
            # ('pipelining-max_loop', self.autcorrelation_extended['pipelining']['max_loop']),
            # ('pipelining-init_zero_loop', self.autcorrelation_extended['pipelining']['init_zero_loop']),
            #('bundling-s', self.autcorrelation_extended['bundling']['s'])
        ]

        # END OF BENCHMARK DEFINITION
        # 最后的benchmark定义

        # 各benchmark的指令
        self.benchmark_directives = {"ChenIDCt": ['column_loops', 'row_loops', 'bundle_b', 'accuracy_loops'],
                                 "Autocorrelation_extended": (
                                 self.autcorrelation_extended, self.autcorrelation_extended_directives_ordered),
                                     "adpcm_decode": ['main_loop_unrolling', 'mac_loop_unrolling',
                                                      'update_loop_unrolling', 'encode_inline',
                                                      'help_function_unrolling', 'bundle_b'],
                                     "adpcm_encode": ['main_loop_unrolling', 'mac_loop_unrolling',
                                                      'update_loop_unrolling', 'encode_inline',
                                                      'help_function_unrolling', 'bundle_b'],
                                     "Reflection_coefficients": ['loop1', 'loop2', 'loop3', 'loop4',
                                                                 'inline_abs', 'inline_norm', 'inline_div',
                                                                 'inline_add', 'inline_mult_r', 'bundle_b'],
                                     "Autocorrelation": ['max_loop', 'gsm_resc_loop', 'init_loop', 'compute_loop',
                                                         'shift_loop', 'bundle_b']}

        # 调用benchmark
        # 输入正确的benchmark 的 name 得到其 synthesis_result configurations, feature_ses , directive
        function_to_call = self.benchmarks_dictionary_data[name]
        if function_to_call is not None:
            self.benchmark_synthesis_results, self.benchmark_configurations, \
            self.benchmark_feature_sets, self.benchmark_directives = function_to_call()
        else:
            self.benchmark_synthesis_results, self.benchmark_configurations, \
            self.benchmark_feature_sets, self.benchmark_directives = None, None, None, None

        print(self.benchmarks_dictionary_data["ChenIDCt"])


    # 得到 Benchmark 的 data
    def get_chenidct_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect('E:\文献管理\文献集\遗传算法\MOEAD-master\src\HLS\datasets\ChenIDct.'+self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute('select latencies, ffs from '+self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select '+directives_str+' from '+self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered

    def get_encode_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect('./datasets/'+self.benchmark_name+'.'+self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute('select latencies, ffs from '+self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select '+directives_str+' from '+self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered

    def get_decode_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect('./datasets/'+self.benchmark_name+'.'+self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute('select latencies, ffs from '+self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select '+directives_str+' from '+self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered

    def get_autocorr_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect('E:\文献管理\文献集\遗传算法\MOEAD-master\src\HLS\datasets\Autocorrelation.'+self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute('select latencies, ffs from '+self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select '+directives_str+' from '+self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered

    def get_reflection_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        # 连接字符串，取到地址值
        conn = sqlite3.connect('./datasets/'+self.benchmark_name+'.'+self.benchmarks_dict_type[self.benchmark_name])
        # 得到综合结果
        synthesis_result = conn.execute('select intervals, ffs from '+self.benchmark_name).fetchall()
        # 得到configuration
        configurations = conn.execute(
            'select '+directives_str+' from '+self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered


    # 定义设计空间
    def _define_desing_space(self, configurations):

        # 特征集
        feature_sets = []
        # 将configurations 中的每个元素 c 转化为列表，并存储在config_list中
        config_list = [list(c) for c in configurations]
        # 将配置列表转换成矩阵matrix
        matrix = np.array(config_list)
        # 获取其shape matrix.shape返回一个元组，元组长度等于其维度 （例如二维，则（a,b） shape[0]为其行数 shape[1]为其列数）
        matrix_shape = matrix.shape

        # 遍历
        for i in range(0, matrix_shape[1]):   # matrix_shape[1] 获取其列数
            # matrix[:, i] 就表示获取 matrix 中所有行（:）的第 i 列元素。这个操作返回的是一个一维数组，包含了 matrix 的第 i 列的所有元素
            column = matrix[:, i]
            # 得到每一列所有取值  每一列对应一条指令
            unique_vector = np.unique(column)
            # unique函数得到的仍然是np格式，因此用to_list将其转换为列表
            tmp = unique_vector.tolist()
            # 将tmp内部的值排序
            tmp.sort()
            # 将其加入到特征集中
            feature_sets.append(tmp)
        # 得到排序后的特征集的索引
        # enumerate(feature_sets) 生成了一个枚举对象，其中包含了 feature_sets 中每个特征集（即列表）的索引和该特征集本身。然后，这个枚举对象被传递给 sorted 函数进行排序。
        # sorted(enumerate(feature_sets), reverse=True, key=lambda x: len(x[1]))：这个函数会根据 key 函数的结果对枚举对象进行排序。在这里，key 函数是 lambda x: len(x[1])，它返回每个元素（即每个特征集）的长度。reverse=True 表示按降序排序，也就是长度最大的特征集排在最前面。
        # 这段代码 [i[0] for i in sorted(...)] 是一个列表推导式，它的作用是遍历 sorted(...) 返回的列表，并从每个元素 i 中取出第一个元素 i[0]。 i[0] 就是索引值
        ordered_sets = [i[0] for i in sorted(enumerate(feature_sets), reverse=True, key=lambda x: len(x[1]))]

        config_reordered = []
        tmp = []

        # 这个for循环有点不懂
        for r in configurations:
            for i in ordered_sets:
                tmp.append(r[i])

            config_reordered.append(tmp)
            tmp = []
        # 排序后特征集
        feature_sets_orderd = [feature_sets[i] for i in ordered_sets]
        # 排序后指令
        directives_ordered = [self.benchmark_directives[self.benchmark_name][i] for i in ordered_sets]

        return config_reordered, feature_sets_orderd, directives_ordered

