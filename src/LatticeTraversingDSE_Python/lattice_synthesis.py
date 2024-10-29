########################################################################################################################
# Author: Lorenzo Ferretti
# Year: 2018
#
# Contains classes Synthesiser and FakeSynthesiser. These are used to invoke Vivado HLS and generate the synthesis data
# given an input configuration. Fake synthesiser retrieve data from the exhaustive exploration already per = numpy.random.beta(a, b, 1)[0]rformed
########################################################################################################################
from os import listdir
from os.path import isfile, join, exists
import subprocess
import xml.etree.ElementTree

class FakeSynthesis:

    def __init__(self, entire_ds, lattice):
        self.entire_ds = entire_ds
        self.lattice = lattice

    def synthesise_configuration(self, config):
        # 原始的指令值的configuration c
        c = self.lattice.revert_discretized_config(config)
        result = None
        # 遍历整个设计空间
        for i in range(0, len(self.entire_ds)):
            # 第i个configuration如果与c
            if self.entire_ds[i].configuration == c:
                # 得到这个configuration对应的latency和area
                result = (self.entire_ds[i].latency, self.entire_ds[i].area)
                break
        return result


class VivdoHLS_Synthesis:

    def __init__(self, lattice, ds_description, ds_description_ordered, ds_bundling, project_description):
        self.lattice = lattice
        self.ds_descriptor = ds_description
        self.ds_descriptor_ordered = ds_description_ordered
        self.bundling_sets = ds_bundling
        self.project_name = project_description["prj_name"]
        self.test_bench = project_description["test_bench_file"]
        self.source_folder = project_description["source_folder"]
        self.top_function = project_description["top_function"]

    def synthesise_configuration(self, config):
        # 原始指令配置
        c = self.lattice.revert_discretized_config(config)
        script_name = self.generate_tcl_script(c)
        if script_name is None:
            return None, None
        # process = subprocess.Popen(["vivado_hls", "-f", "./exploration_scripts/" + script_name + ".txt", ">>", script_name + ".out"])
        if exists("./"+self.project_name+"/"+script_name+"/syn/report/"+self.top_function+"_csynth.xml"):
            print("File already synthesised and already in folder!")
            pass
        else:
            process = subprocess.Popen("vivado_hls -f ./exploration_scripts/" + script_name + ".txt >> ./exploration_scripts/" + script_name + ".out", shell=True)
            process.wait()
        latency, area = self.get_synthesis_results(script_name)
        return latency, area

    # 产生脚本
    def generate_tcl_script(self, configuration):
        # 得到时钟周期
        clock = self.ds_descriptor["clock"]
        # 换行
        new_line = " \n"
        script = "open_project " + self.project_name + new_line
        script += "set_top " + self.top_function + new_line
        file_list = []
        test_bench = None
        # 遍历文件夹内所有文件和文件夹
        for f in listdir(self.source_folder):
            # 这行代码检查当前项（由源文件夹路径和当前项名称连接而成的路径）是否是一个常规文件。
            if isfile(join(self.source_folder, f)):
                if f != self.test_bench:
                    # 当前文件不是测试的bench，检查当前文件的扩展名是否为.c或.h。
                    if f[-2:] == ".c" or f[-2:] == ".h":
                        # 添加到文件列表中
                        file_list.append(f)
                else:
                    # 说明是测试的bench，添加到test_bench中
                    test_bench = f
        # 测试台文件
        script += "add_files -tb " + self.source_folder + '/' + test_bench + new_line
        for f in file_list:
            # 遍历文件列表
            script += "add_files " + self.source_folder + '/' + f + new_line

        # 所有配置参数
        script += "open_solution sol_" + "_".join(str(e) for e in configuration) + new_line
        # fpga型号为：xc7k160tfbg484-1
        script += "set_part {xc7k160tfbg484-1}" + new_line
        if isinstance(clock, dict):
            # 如果clock 是 dict 的子类
            clock_list = clock["clock"]

            # 这行代码是一个列表推导式，遍历self.ds_descriptor_ordered每个元素（是一个元组），检查每个元组的第1个元素tupl[0]是否等于”clock-clock“
            # 如果等于，将该元组的索引更新到列表clock_idx 中
            clock_idx = [i for i, tupl in enumerate(self.ds_descriptor_ordered) if tupl[0] == "clock-clock"]
            # clock_idx = self.ds_descriptor.benchmark_directives.index("clock")
            # 创建了一个时钟，周期为 configuration[clock_idx[0]]
            script += "create_clock -period " + str(configuration[clock_idx[0]]) + " -name default" + new_line
        else:
            # 如果clock不是一个字典，使用默认10ns作为时钟周期
            script += "create_clock -period 10 -name default" + new_line

        # script += "set_directive_interface -mode s_axilite \"" + self.top_function + "\"" + new_line

        # Start exploring all the other configuration except the clock
        # 探索除时钟周期以外的其他所有配置

        # 遍历configuration中的每个元素
        for c in range(len(configuration)):
            if c == clock_idx[0]:
                # 当前索引等于时钟索引，说明满足情况，跳出。
                continue
            else:
                # 如果当前索引不等于时钟索引
                # 获取当前configuration
                conf = configuration[c]
                # 获取对应指令
                directive = self.ds_descriptor_ordered[c]
                # 将指令添加到脚本中
                directive = self.add_directive(conf, directive, script)
                # 如果不存在指令，返回none
                if directive is None:
                    return None
                # 将指令添加到脚本当中
                script += directive

        # 该命令用于启动综合设计过程
        script += "csynth_design" + new_line
        # 退出当前环境或程序
        script += "exit" + new_line

        # 创建文件名，文件名由sol_和配置参数组成，配置参数之间
        script_file = "sol_" + "_".join(str(x) for x in configuration)
        # 创建输出脚本文件 "w"代表write，如果文件已存在，内容将被清空，不存在，则创建一个新文件。
        outfile = open("./exploration_scripts/"+script_file+".txt", "w")
        # 将上述代码添加的script添加到输出文件中
        outfile.write(script)
        outfile.close()
        # 返回脚本文件
        return script_file

    # 添加指令
    def add_directive(self, directive_value, directive, script):
        # 取出字符串第一个元素，得到指令类型
        kind = directive[0].split('-')[0]
        # 分析指令类型，做出对应操作
        if kind == "unrolling":
            script = self.add_unrolling_directive(directive_value, directive)
        if kind == "bundling":
            script = self.add_bundling_directive(directive_value)
        if kind == "pipelining":
            script = self.add_pipeline_directive(directive_value, directive, script)
            if script is None:
                return None
        if kind == "inlining":
            script = self.add_inlining_directive(directive_value, directive)
        if kind == "partitioning":
            script = self.add_partitioning_directive(directive_value, directive)

        return script

    def add_unrolling_directive(self, directive_value, directive):
        new_line = "\n"
        # 得到指令中的loop_name
        loop_name = directive[0].split('-')[1]
        script = ""
        if directive_value != 0:
            script += "set_directive_unroll -factor " + str(directive_value) + " \"" + self.top_function + "/" + loop_name \
                      + "\"" + new_line
        else:
            script += "set_directive_loop_flatten -off \"" + self.top_function + "/" + loop_name + "\"" + new_line

        return script

    def add_bundling_directive(self, directive_value):
        new_line = "\n"
        script = ""
        # 添加一个接口模式的指令，将顶层函数的接口模式设置为s_axilite
        # s_axilite：这是一种轻量级的 AXI 接口协议，通常用于低吞吐量的控制寄存器访问。例如，你可以使用 s_axilite 接口来控制 IP 核的启动和停止操作。
        # 在 HLS 中，你可以使用 #pragma HLS INTERFACE s_axilite port=return 指令来设置类型为 s_axilite 的函数实参，端口名称“return”将在IP块中创建中断信号。
        script += "set_directive_interface -mode s_axilite \"" + self.top_function + "\"" + new_line
        # 获得绑定集的端口
        bundle_ports = self.bundling_sets[0]
        # 获取与指令值对应的绑定集
        bundle_sets = self.bundling_sets[1][directive_value]
        for i in range(0, len(bundle_ports)):
            # 遍历所有端口
            # 将端口设置为m_axi 用于高吞吐量数据传输
            script += "set_directive_interface -mode m_axi -offset direct -bundle " + str(bundle_sets[i]) + " \"" +\
                      self.top_function + "\" " + bundle_ports[i] + new_line
        return script

    # 添加流水线指令
    def add_pipeline_directive(self, directive_value, directive, old_script):
        new_line = "\n"
        script = ""
        target = directive[0].split('-')[1]
        # 从old_script中找到最近的target，返回索引
        idx = old_script.find(target)
        if idx >= 0:
            # 存在于target相同的值
            # 获取从idx-30 到 index的子字符串
            substring = old_script[idx-30:idx]
            # 从substring中分裂每个字符，将只是数字部分的字符存储在unroll列表中
            # isdigit() 检查字符串是否只由数字组成
            unroll = [int(s) for s in substring.split() if s.isdigit()]
            if len(unroll) != 0:
                # 弹出最后一个元素
                unroll = unroll.pop()
                if unroll == 9 or unroll == 160 or unroll == 152:
                    # 等于9， 160， 152就返回none
                    return None

        # 存在循环
        if target.find("loop") >= 0:
            if directive_value != 0:
                # 指令值不为0
                script += 'set_directive_pipeline \"' + self.top_function + "/" + target + '\"\n'
            else:
                # 指令为0 关闭流水线
                script += "set_directive_pipeline -off " + "\"" + self.top_function + "/" + target + "\"" + new_line
        else:
            # 不存在循环
            if directive_value != 0:
                # 根据对应的值添加对应的pipeline指令
                script += 'set_directive_pipeline \"' + target + '\"\n'
            else:
                # 指令为0 关闭流水线
                script += "set_directive_pipeline -off " + "\"" + target + "\"" + new_line

        return script

    # 内联指令
    def add_inlining_directive(self, directive_value, directive):
        new_line = "\n"
        script = ""
        # 得到taget，通过分裂取出第二个元素实现（索引为1）
        target = directive[0].split('-')[1]

        if directive_value == 0:
            # 指令值等于零
            # 关闭内联
            script = script + "set_directive_inline -off \"" + target + "\"" + new_line
        else:
            # 不等于零，添加内联指令
            script = script + "set_directive_inline \"" + target + "\"" + new_line

        return script

    def add_partitioning_directive(self, directive_value, directive):
        new_line = "\n"
        script = ""
        target = directive[0].split('-')[1]
        type = "block"
        if directive_value != 0:
            script += "set_directive_array_partition -type " + type + " -factor " + str(directive_value) + " -dim 1 \""\
                      + self.top_function + "\" " + target + new_line

        return script

    # 得到综合结果
    def get_synthesis_results(self, script_name):
        # 打开脚本输出文件 line139创建， 读模式
        outputfile = open("./exploration_scripts/"+script_name+".out", "r")
        # 将文件内容放入content中
        content = outputfile.read()

        # 检查content中是否包含 has been removed
        if content.find("has been removed") > 0:
            # 如有 将参数设置为一个较大值
            x = 100000
            LUT = 100000
            FF = 1000000
        else:
            # 如没有，将参数设置为none
            x = None
            LUT = None
            FF = None

            # 使用xml.etree.ElementTree模块解析xml文件，先构造xml文件的路径，再用.parse函数解析地址，返回一个ElementTree对象，用.getroot()得到xml树的根元素
            root = xml.etree.ElementTree.parse("./"+self.project_name+"/"+script_name+"/syn/report/"+self.top_function+"_csynth.xml").getroot()
            # For each solution extract Area estimation
            # 遍历根元素下所有AreaEstimates结点
            for area_est in root.findall('AreaEstimates'):
                # 对每个AreaEstimates结点，遍历所有的resource结点
                for resource in area_est.findall('Resources'):
                    # 对于每个resource结点，遍历每个孩子节点
                    for child in resource:
                        # 检查孩子节点的标签中是否包含 FF LUT
                        if "FF" in child.tag:
                            # 将文本内容转化成整数
                            FF = int(child.text)

                        if "LUT" in child.tag:
                            # 将文本内容转化成整数
                            LUT = int(child.text)

            # For each solution extract Performance estimation
            for perf_est in root.findall('PerformanceEstimates'):
                for latency in perf_est.findall('SummaryOfOverallLatency'):
                    for child in latency:
                        if "Average-caseLatency" in child.tag:
                            x = int(child.text)

        latency = x
        area = LUT
        return latency, area
