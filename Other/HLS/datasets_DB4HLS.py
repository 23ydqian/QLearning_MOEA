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
        self.benchmarks_dict_type = {"aes_addRoundKey_aes_aes": "db",
                                     "aes_addRoundKey_cpy_aes_aes": "db",
                                     "aes_expandEncKey_aes_aes": 'db',
                                     "aes256_encrypt_ecb_aes_aes": "db",
                                     "aes_notable": "db",
                                     "aes_table": "db",
                                     "ncubed": "db",
                                     "stencil2d": "db",
                                     "stencil_stencil2d_stencil": "db",
                                     "radix_local_scan": "db",
                                     "ss_sort_radix_sort": "db",

                                     "add_bias_to_activations": "db",
                                     "backprop_backprop_backprop": "db",
                                     "get_delta_matrix_weights1": "db",
                                     "get_delta_matrix_weights2": "db",
                                     "get_delta_matrix_weights2_backprop_backprop": "db",
                                     "get_delta_matrix_weights3": "db",
                                     "get_oracle_activations1_backprop_backprop": "db",
                                     "get_oracle_activations2_backprop_backprop": "db",
                                     "matrix_vector_product_with_bias_input_layer": "db",
                                     "matrix_vector_product_with_bias_output_layer": "db",
                                     "matrix_vector_product_with_bias_second_layer": "db",
                                     "soft_max_backprop_backprop": "db",
                                     "take_difference_backprop_backprop": "db",
                                     "update_weights_backprop_backprop": "db",

                                     "bulk": "db",
                                     "hist_radix_sort": "db",
                                     "init_radix_sort": "db",
                                     "last_step_scan_radix_sort": "db",
                                     "local_scan_radix_sort": "db",
                                     "sum_scan_radix_sort": "db",
                                     "update_radix_sort": "db",

                                     "fft_strided_fft": "db",
                                     "md_kernel_knn_md": "db",
                                     "stencil3d_stencil3d_stencil": "db",
                                     "viterbi_viterbi_viterbi": "db",
                                     "bbgemm_blocked_gemm": "db",
                                     "ellpack_ellpack_spmv": "db",
                                     "merge_merge_sort": "db",
                                     "ms_mergesort_merge_sort": "db",}

        self.benchmarks_dictionary_data = {"aes_addRoundKey_cpy_aes_aes": self.get_aes_addRoundKey_cpy_aes_aes_data,
                                           "aes_addRoundKey_aes_aes": self.get_aes_addRoundKey_aes_aes_data,
                                           "aes_expandEncKey_aes_aes": self.get_aes_expandEncKey_aes_aes_data,
                                           "aes256_encrypt_ecb_aes_aes": self.get_aes256_encrypt_ecb_aes_aes_data,
                                           "aes_notable": self.get_aes_notable_data,
                                           "aes_table": self.get_aes_table_data,
                                           "ncubed": self.get_ncubed_data,
                                           "stencil2d": self.get_stencil2d_data,
                                           "stencil_stencil2d_stencil": self.get_stencil_stencil2d_stencil_data,
                                           "radix_local_scan": self.get_radix_local_scan_data,
                                           "ss_sort_radix_sort": self.get_ss_sort_radix_sort_data,
                                           "add_bias_to_activations": self.get_add_bias_to_activations_data,
                                           "backprop_backprop_backprop": self.get_backprop_backprop_backprop_data,
                                           "get_delta_matrix_weights1": self.get_get_delta_matrix_weights1_data,
                                           "get_delta_matrix_weights2": self.get_get_delta_matrix_weights2_data,
                                           "get_delta_matrix_weights2_backprop_backprop": self.get_get_delta_matrix_weights2_backprop_backprop_data,
                                           "get_delta_matrix_weights3": self.get_get_delta_matrix_weights3_data,
                                           "get_oracle_activations1_backprop_backprop": self.get_get_oracle_activations1_backprop_backprop_data,
                                           "get_oracle_activations2_backprop_backprop": self.get_get_oracle_activations2_backprop_backprop_data,
                                           "matrix_vector_product_with_bias_input_layer": self.get_matrix_vector_product_with_bias_input_layer_data,
                                           "matrix_vector_product_with_bias_output_layer": self.get_matrix_vector_product_with_bias_output_layer_data,
                                           "matrix_vector_product_with_bias_second_layer": self.get_matrix_vector_product_with_bias_second_layer_data,
                                           "soft_max_backprop_backprop": self.get_soft_max_backprop_backprop_data,
                                           "take_difference_backprop_backprop": self.get_take_difference_backprop_backprop_data,
                                           "update_weights_backprop_backprop": self.get_update_weights_backprop_backprop_data,
                                           "bulk": self.get_bulk_data,
                                           "hist_radix_sort": self.get_hist_radix_sort_data,
                                           "init_radix_sort": self.get_init_radix_sort_data,
                                           "last_step_scan_radix_sort": self.get_last_step_scan_radix_sort_data,
                                           "local_scan_radix_sort": self.get_local_scan_radix_sort_data,
                                           "sum_scan_radix_sort": self.get_sum_scan_radix_sort_data,
                                           "update_radix_sort": self.get_update_radix_sort_data,
                                           "fft_strided_fft": self.get_fft_strided_fft_data,
                                           "md_kernel_knn_md": self.get_md_kernel_knn_md_data,
                                           "stencil3d_stencil3d_stencil": self.get_stencil3d_stencil3d_stencil_data,
                                           "viterbi_viterbi_viterbi": self.get_viterbi_viterbi_viterbi_data,
                                           "bbgemm_blocked_gemm": self.get_bbgemm_blocked_gemm_data,
                                           "ellpack_ellpack_spmv": self.get_ellpack_ellpack_spmv_data,
                                           "merge_merge_sort": self.get_merge_merge_sort_data,
                                           "ms_mergesort_merge_sort": self.get_ms_mergesort_merge_sort_data,
                                            }


        self.benchmark_directives = {"aes_addRoundKey_aes_aes": ['array_partition_buf','array_partition_key','unroll'],
                                     "aes_addRoundKey_cpy_aes_aes": ['array_partition_buf','array_partition_key', 'array_partition_cpk', 'unroll'],
                                     "aes_expandEncKey_aes_aes": ['core', 'array_partition', 'unroll4', 'unroll1'],
                                     "aes256_encrypt_ecb_aes_aes": ['array_partition_ctx', 'array_partition_k', 'unroll_ecb2', 'unroll_ecb3', 'inline1', 'inline2','inline3'],
                                     "aes_notable": ['resource_ctk', 'resource_k', 'resource_buf', 'array_partition', 'unrolling_ecb1', 'unrolling_ecb2','unrolling_ecb3',
                                                     'inlining1', 'inlining2','inlining3'],
                                     "aes_table": ['resource_ctk', 'resource_k', 'resource_buf', 'array_partition', 'unrolling_ecb', 'unrolling_ecb2','unrolling_ecb3','inlining'],
                                     "ncubed": ['array_partition_m1', 'array_partition_m2', 'array_partition_prod', 'unroll_outer', 'unroll_middle', 'unroll_inner'],
                                     "stencil2d": ['array_partition_orig', 'array_partition_sol', 'unroll_label1', 'unroll_label2'],
                                     "stencil_stencil2d_stencil": ['array_partition_filter', 'unroll_label1', 'unroll_label2'],
                                     "radix_local_scan": ['array_partition', 'unroll_local1', 'unroll_local2', 'clock'],
                                     "ss_sort_radix_sort": ['array_partition', 'unroll', 'inline_init', 'inline_hist', 'inline_local_scan', 'inline_sum_scan','inline_last_step_scan','inline_update'],
                                     "add_bias_to_activations": ['array_partition_biases', 'array_partition_activations', 'unroll'],
                                     "backprop_backprop_backprop": ['array_partition1', 'array_partition2', 'array_partition3', 'array_partition4', 'array_partition5', 'array_partition6', 'array_partition7', 'array_partition8', 'array_partition9', 'array_partition10', 'unroll'],
                                     "get_delta_matrix_weights1": ['array_partition_delta_weights1', 'array_partition_output_difference', 'array_partition_last_activations','unroll1', 'unroll2'],
                                     "get_delta_matrix_weights2": ['array_partition_delta_weights2', 'array_partition_output_difference', 'array_partition_last_activations','unroll1', 'unroll2'],
                                     "get_delta_matrix_weights2_backprop_backprop": ['array_partition','unroll1', 'unroll2'],
                                     "get_delta_matrix_weights3": ['array_partition_delta_weights3', 'array_partition_output_difference', 'array_partition_last_activations','unroll1', 'unroll2'],
                                     "get_oracle_activations1_backprop_backprop": ['array_partition_weights2', 'array_partition_dactivations','unroll1', 'unroll2'],
                                     "get_oracle_activations2_backprop_backprop": ['array_partition_weights3', 'array_partition_output_differences', 'array_partition_dactivations','unroll1', 'unroll2'],
                                     "matrix_vector_product_with_bias_input_layer": ['array_partition_biases', 'array_partition_weights','unroll1', 'unroll2','inline'],
                                     "matrix_vector_product_with_bias_output_layer": ['array_partition_weights', 'array_partition_activations','unroll1', 'unroll2','inline'],
                                     "matrix_vector_product_with_bias_second_layer": ['array_partition_biases', 'array_partition_weights','unroll','inline'],
                                     "soft_max_backprop_backprop": ['array_partition_net_outputs', 'array_partition_activations','unroll1', 'unroll2'],
                                     "take_difference_backprop_backprop": ['array_partition_net_outputs', 'array_partition_solutions','array_partition_output_difference', 'array_partition_dactivations','unroll'],
                                     "update_weights_backprop_backprop": ['array_partition_weights1', 'array_partition_weights3','array_partition_d_weights1', 'array_partition_d_weights3','array_partition_biases1', 'array_partition_biases3','array_partition_d_biases1', 'array_partition_d_biases3','unroll1','unroll2'],

                                     "bulk": ['nodes', 'level','array_partition_nodes', 'array_partition_level','unroll'],
                                     "hist_radix_sort": ['array_partition_bucket', 'array_partition_a','unroll1', 'unroll2'],
                                     "init_radix_sort": ['resource', 'array_partition','unroll'],
                                     "last_step_scan_radix_sort": ['array_partition_bucket', 'array_partition_sum','unroll'],
                                     "local_scan_radix_sort": ['array_partition', 'unroll1','unroll2'],
                                     "sum_scan_radix_sort": ['resource_sum', 'resource_bucket','array_partition_sum', 'array_partition_bucket'],
                                     "update_radix_sort": ['array_partition_b', 'array_partition_bucket','array_partition_a', 'unroll1','unroll2'],

                                     "fft_strided_fft": ['array_partition_real', 'array_partition_img','array_partition_real_twid','array_partition_img_twid', 'unroll1','unroll2'],
                                     "md_kernel_knn_md": ['array_partition_force', 'array_partition_position','array_partition_NL', 'unroll1','unroll2'],
                                     "stencil3d_stencil3d_stencil": ['array_partition_orig', 'array_partition_sol', 'unroll1','unroll2', 'unroll3','unroll4', 'unroll5'],
                                     "viterbi_viterbi_viterbi": ['array_partition_obs', 'array_partition_init','array_partition_transition', 'array_partition_emission', 'array_partition_path','array_partition_llike', 'unroll1','unroll2', 'unroll3'],
                                     "bbgemm_blocked_gemm": ['array_partition_m1', 'array_partition_m2', 'unroll1','unroll2', 'unroll3','unroll4'],
                                     "ellpack_ellpack_spmv": ['array_partition_nzval', 'array_partition_vec','array_partition_out', 'unroll1','unroll2'],
                                     "merge_merge_sort": ['array_partition', 'unroll1','unroll2', 'unroll3'],
                                     "ms_mergesort_merge_sort": ['resource', 'array_partition', 'unroll1','unroll2','inline',],}

        function_to_call = self.benchmarks_dictionary_data[name]
        if function_to_call is not None:
            self.benchmark_synthesis_results, self.benchmark_configurations, \
            self.benchmark_feature_sets, self.benchmark_directives = function_to_call()
        else:
            self.benchmark_synthesis_results, self.benchmark_configurations, \
            self.benchmark_feature_sets, self.benchmark_directives = None, None, None, None

    def get_ms_mergesort_merge_sort_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)
        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_merge_merge_sort_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)
        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_ellpack_ellpack_spmv_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)
        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_bbgemm_blocked_gemm_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)
        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_viterbi_viterbi_viterbi_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)
        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_stencil3d_stencil3d_stencil_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)
        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_md_kernel_knn_md_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_fft_strided_fft_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_update_radix_sort_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_sum_scan_radix_sort_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_local_scan_radix_sort_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_last_step_scan_radix_sort_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_init_radix_sort_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_hist_radix_sort_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_bulk_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_update_weights_backprop_backprop_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_take_difference_backprop_backprop_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_soft_max_backprop_backprop_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_matrix_vector_product_with_bias_second_layer_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_matrix_vector_product_with_bias_output_layer_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_matrix_vector_product_with_bias_input_layer_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_get_oracle_activations2_backprop_backprop_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_get_oracle_activations1_backprop_backprop_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_get_delta_matrix_weights3_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_get_delta_matrix_weights2_backprop_backprop_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_get_delta_matrix_weights2_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_get_delta_matrix_weights1_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_backprop_backprop_backprop_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_add_bias_to_activations_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_ss_sort_radix_sort_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered
    def get_radix_local_scan_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered

    def get_stencil_stencil2d_stencil_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect(
            './datasets/' + self.benchmark_name + '.' + self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute(
            'select average_latency, ffs, hls_execution_time from ' + self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select ' + directives_str + ' from ' + self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered

    def get_stencil2d_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect('./datasets/'+self.benchmark_name+'.'+self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute('select average_latency, ffs, hls_execution_time from '+self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select '+directives_str+' from '+self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered

    def get_ncubed_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect('./datasets/'+self.benchmark_name+'.'+self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute('select average_latency, ffs, hls_execution_time from '+self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select '+directives_str+' from '+self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered

    def get_aes_table_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect('./datasets/'+self.benchmark_name+'.'+self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute('select average_latency, ffs, hls_execution_time from '+self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select '+directives_str+' from '+self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered


    def get_aes_addRoundKey_aes_aes_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect('./datasets/'+self.benchmark_name+'.'+self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute('select average_latency, ffs, hls_execution_time from '+self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select '+directives_str+' from '+self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered

    def get_aes_addRoundKey_cpy_aes_aes_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect('./datasets/'+self.benchmark_name+'.'+self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute('select average_latency, ffs, hls_execution_time from '+self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select '+directives_str+' from '+self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered

    def get_aes_expandEncKey_aes_aes_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect('./datasets/'+self.benchmark_name+'.'+self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute('select average_latency, ffs, hls_execution_time from '+self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select '+directives_str+' from '+self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered

    def get_aes256_encrypt_ecb_aes_aes_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect('./datasets/'+self.benchmark_name+'.'+self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute('select average_latency, ffs, hls_execution_time from '+self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select '+directives_str+' from '+self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered

    def get_aes_notable_data(self):
        directives_str = ",".join(self.benchmark_directives[self.benchmark_name])
        conn = sqlite3.connect('./datasets/'+self.benchmark_name+'.'+self.benchmarks_dict_type[self.benchmark_name])
        synthesis_result = conn.execute('select average_latency, ffs, hls_execution_time from '+self.benchmark_name).fetchall()
        configurations = conn.execute(
            'select '+directives_str+' from '+self.benchmark_name).fetchall()

        # Define DS
        config_reordered, feature_sets_orderd, directives_ordered = self._define_desing_space(configurations)

        return synthesis_result, config_reordered, feature_sets_orderd, directives_ordered




    def _define_desing_space(self, configurations):

        feature_sets = []
        config_list = [list(c) for c in configurations]
        matrix = np.array(config_list)
        matrix_shape = matrix.shape
        for i in range(0, matrix_shape[1]):
            column = matrix[:, i]
            unique_vector = np.unique(column)
            tmp = unique_vector.tolist()
            tmp.sort()
            feature_sets.append(tmp)
        ordered_sets = [i[0] for i in sorted(enumerate(feature_sets), reverse=True, key=lambda x: len(x[1]))]
        config_reordered = []
        tmp = []
        for r in configurations:
            for i in ordered_sets:
                tmp.append(r[i])

            config_reordered.append(tmp)
            tmp = []
        feature_sets_orderd = [feature_sets[i] for i in ordered_sets]
        directives_ordered = [self.benchmark_directives[self.benchmark_name][i] for i in ordered_sets]

        return config_reordered, feature_sets_orderd, directives_ordered
