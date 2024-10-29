import datasets
from sythesis import FakeSynthesis
from DSpoint import DSpoint
import ADRSunit
import numpy as np

# 具体看benchmark
Dimention = 4

Func_num = 2
Bound = [0, 10000000]

def Func(config, entire_ds):
    f1 = area(config, entire_ds)
    f2 = latency(config, entire_ds)
    return [f1, f2]

def area(config,entire_ds):
    # 引入指令、configuration、area、latency
    # benchmark = ["ChenIDCt", "adpcm_decode", "adpcm_encode", "Autocorrelation", "Reflection_coefficients"]
    hls = FakeSynthesis(entire_ds)
    area = hls.synthesise_configuration(config)[1]
    return area

def latency(config,entire_ds):
    # 引入指令、configuration、area、latency
    # benchmark = ["ChenIDCt", "adpcm_decode", "adpcm_encode", "Autocorrelation", "Reflection_coefficients"]
    hls = FakeSynthesis(entire_ds)
    latency = hls.synthesise_configuration(config)[0]
    return latency

# 归一化函数
def Normalize(res, synthesis_result):
    max_latency = max(x[0] for x in synthesis_result)
    min_latency = min(x[0] for x in synthesis_result)
    max_area = max(x[1] for x in synthesis_result)
    min_area = min(x[1] for x in synthesis_result)
