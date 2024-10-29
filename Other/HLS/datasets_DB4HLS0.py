import sqlite3
import numpy as np

class Datasets_DB4HLS:
    # 资源总数
    LUT_total = 274080
    FF_total = 548160
    DSP_total = 2520
    BRAM_total = 1840

    def __init__(self, name):
        self.benchmark_name = name

        # self.benchmarks_dictionary_data = {
        #     "'ellpack_ellpack_spmv'": self.get_ellpack_data,
        #     "'bulk'": self.get_data,  # 这个benchmark缺了两个点
        #     "'md_kernel_knn_md'": self.get_data
        # }

        self.benchmarks_dictionary_features = {
            "'ellpack_ellpack_spmv'": [4, 6, 7, 8, 9],
            "'bulk'": [0, 2, 4, 6, 8], # 这个benchmark缺了两个点
            "'md_kernel_knn_md'": [7, 10, 13, 14, 15],
            "'viterbi_viterbi_viterbi'": [5, 6, 7, 8, 9, 10, 11, 12, 13],
            "'bbgemm_blocked_gemm'": [3, 4, 6, 7, 8, 9],
            "'ncubed'" : [0, 1, 2, 3, 4],  # 有很多null值
            "'merge_merge_sort'": [2, 3, 4, 5],
            "'ms_mergesort_merge_sort'": [0, 1, 2, 3, 4],
            "'stencil3d_stencil3d_stencil'": [4, 5, 6, 7, 12, 13, 14],
            "'fft_strided_fft'": [4, 5, 6, 7, 8],
            # backprop
            "'get_delta_matrix_weights1'": [0, 1, 2, 3, 4],
            "'get_delta_matrix_weights2'": [0, 1, 2, 3, 4],
            "'get_delta_matrix_weights3'": [0, 1, 2, 3, 4],
            "'get_oracle_activations1_backprop_backprop'": [6, 7, 8, 9],
            "'get_oracle_activations2_backprop_backprop'": [4, 5, 7, 8, 9],
            "'matrix_vector_product_with_bias_input_layer'": [4, 5, 7, 8, 10],
            "'matrix_vector_product_with_bias_second_layer'": [0, 1, 4, 6],
            "'matrix_vector_product_with_bias_output_layer'": [0, 1, 2, 5, 6],
            "'backprop_backprop_backprop'": [0, 3, 5, 7, 8, 10, 14, 15, 16, 19, 22],
            "'add_bias_to_activations_backprop_backprop'": [2, 3, 4],
            "'soft_max_backprop_backprop'": [0, 1, 2, 3],
            "'take_difference_backprop_backprop'": [0, 1, 2, 3, 4],
            "'update_weights_backprop_backprop'": [0, 2, 3, 5, 6, 8, 9, 11, 13, 14],
            ## radix sort
            "'update_radix_sort'": [3, 4, 5, 6, 7],
            "'hist_radix_sort'": [2, 3, 4, 5],
            "'init_radix_sort'": [0, 1, 2],
            "'sum_scan_radix_sort'": [0, 1, 2, 3],
            "'last_step_scan_radix_sort'": [2, 3, 5],
            "'local_scan_radix_sort'": [1, 2, 3],
            "'ss_sort_radix_sort'": [4, 8, 9, 10, 11, 12, 13, 14],
            ## aes
            "'aes_addRoundKey_aes_aes'": [0, 1, 2],
            "'aes_subBytes_aes_aes'": [0, 1],
            "'aes_addRoundKey_cpy_aes_aes'": [0, 1, 2, 3],
            "'aes_shiftRows_aes_aes'": [0, 1],
            "'aes_mixColumns_aes_aes'": [0, 1],
            "'aes_expandEncKey_aes_aes'": [0, 1, 2, 3],
            "'aes256_encrypt_ecb_aes_aes'": [3, 6, 7, 9, 10, 11, 12, 13]



        }

        function_to_call = self.get_data
        if function_to_call is not None:
            self.benchmark_synthesis_results, self.benchmark_configurations, \
                self.benchmark_feature_sets = function_to_call()
        else:
            self.benchmark_synthesis_results, self.benchmark_configurations, \
                self.benchmark_feature_sets = None, None, None

    def get_data(self):
        # 调用数据库
        conn = sqlite3.connect('C:/Users/19519/Desktop/datasets/SQLITE_DB.db')
        cur = conn.cursor()

        # 编写查询语句得到synthesis_result, configurations, entire_ds,处理异常值节点
        sql_text1 = "select config from synthesis_result_new where synthesis_result_new.name = " + self.benchmark_name
        sql_text2 = "select average_latency, (hls_ff + hls_lut + hls_bram + hls_dsp)/4 AS area from synthesis_result_new where synthesis_result_new.name = " + self.benchmark_name
        sql_text3 = "select hls_exit_value from synthesis_result_new where synthesis_result_new.name = " + self.benchmark_name

        config = cur.execute(sql_text1).fetchall()
        processed_config = [eval(item[0]) for item in config]
        config_list = [list(c) for c in processed_config]
        synthesis_result_orgin = cur.execute(sql_text2).fetchall()
        synthesis_result = [list(result) for result in synthesis_result_orgin]
        hls_exit_value = cur.execute(sql_text3).fetchall()
        exit_list = [item[0] for item in hls_exit_value]

        # 将失败综合点对应的synthesis_result 赋值 999999，999999
        for i in range(0, len(exit_list)):
            if exit_list[i] != 0:
                synthesis_result[i][0] = 6666
                synthesis_result[i][1] = 6666

        # feature_sets = [[RAM_1P_BRAM,RAM_2P_BRAM], [RAM_1P_BRAM,RAM_2P_BRAM], [(cyclic,block)(1,2,8,32,64,128,256)], [(cyclic,block)(1,2,8,32,64,128,256)], [1,5,10]] 理论feature_set
        feature_sets = []
        matrix = np.array(config_list, dtype=object)
        columns_to_keep = self.benchmarks_dictionary_features[self.benchmark_name]
        matrix_filtered = matrix[:, columns_to_keep]
        matrix_shape = matrix_filtered.shape
        for i in range(0, matrix_shape[1]):
            column = matrix_filtered[:, i]
            unique_vector = np.unique(column)
            tmp = unique_vector.tolist()
            tmp.sort()
            feature_sets.append(tmp)

        configurations = matrix_filtered.tolist()
        # # fft_strided_fft
        # configurations = [result[:1] + [result[1][0]] + [result[2][0]] + [result[3][0]] + result[4:] for result in configurations]

        return synthesis_result, configurations, feature_sets

    # 用于特定benchmark
    def get_data_special(self):
        # 调用数据库
        conn = sqlite3.connect('C:/Users/Young/Desktop/SQLITE_DB.db')
        cur = conn.cursor()

        # 编写查询语句得到synthesis_result, configurations, entire_ds,处理异常值节点
        sql_text1 = "select config from synthesis_result_new where synthesis_result_new.name = " + self.benchmark_name
        sql_text2 = "select average_latency, (hls_ff + hls_lut + hls_bram + hls_dsp)/4 AS area from synthesis_result_new where synthesis_result_new.name = " + self.benchmark_name
        sql_text3 = "select hls_exit_value from synthesis_result_new where synthesis_result_new.name = " + self.benchmark_name

        config = cur.execute(sql_text1).fetchall()
        processed_config = [eval(item[0]) for item in config]
        config_list = [list(c) for c in processed_config]
        synthesis_result_orgin = cur.execute(sql_text2).fetchall()
        synthesis_result = [list(result) for result in synthesis_result_orgin]
        hls_exit_value = cur.execute(sql_text3).fetchall()
        exit_list = [item[0] for item in hls_exit_value]

        # 将失败综合点对应的synthesis_result 赋值 999999，999999
        for i in range(0, len(exit_list)):
            if exit_list[i] != 0:
                synthesis_result[i][0] = 39999
                synthesis_result[i][1] = 39999

        # feature_sets = [[RAM_1P_BRAM,RAM_2P_BRAM], [RAM_1P_BRAM,RAM_2P_BRAM], [(cyclic,block)(1,2,8,32,64,128,256)], [(cyclic,block)(1,2,8,32,64,128,256)], [1,5,10]] 理论feature_set
        feature_sets = []
        matrix = np.array(config_list, dtype=object)
        columns_to_keep = self.benchmarks_dictionary_features[self.benchmark_name]
        matrix_filtered = matrix[:, columns_to_keep]
        configurations = matrix_filtered.tolist()
        # # fft
        # final_config = [result[:1] + [result[1][0]] + [result[2][0]] + [result[3][0]] + result[4:] for result in
        #                 configurations]
        # undate
        final_config = [result[:1] + [result[1][0]] + [result[2][0]] + result[3:] for result in
                        configurations]
        configurations = final_config
        matrix_filtered = np.array(final_config, dtype=object)
        matrix_shape = matrix_filtered.shape
        for i in range(0, matrix_shape[1]):
            column = matrix_filtered[:, i]
            unique_vector = np.unique(column)
            tmp = unique_vector.tolist()
            tmp.sort()
            feature_sets.append(tmp)
        # # fft_strided_fft
        # configurations = [result[:1] + [result[1][0]] + [result[2][0]] + [result[3][0]] + result[4:] for result in configurations]

        return synthesis_result, configurations, feature_sets





