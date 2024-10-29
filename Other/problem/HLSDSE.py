import datasets
from Other.HLS.sythesis import FakeSynthesis
from Other.HLS.DSpoint import DSpoint
import Other.HLS.ADRSunit as ADRSunit
import numpy as np

# 具体看benchmark
Dimention = 4

Func_num = 2
Bound = [0, 10000000]

def Func(config, moead):
    entire_ds = moead.entire_ds
    f1 = area(config, entire_ds)
    f2 = latency(config, entire_ds)
    # 归一化
    res = [f1,f2]
    Normalize_result(res,moead)
    return res

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

def time(config,entire_ds):
    # 引入指令、configuration、area、latency
    # benchmark = ["ChenIDCt", "adpcm_decode", "adpcm_encode", "Autocorrelation", "Reflection_coefficients"]
    hls = FakeSynthesis(entire_ds)
    time = hls.synthesise_configuration(config)[2]
    return time

def Normalize_result(res, moead):
    max_area = moead.max_area
    min_area = moead.min_area
    max_latency = moead.max_latency
    min_latency = moead.min_latency
    res[0] = (res[0] - min_area) / (max_area - min_area)
    res[1] = (res[1] - min_latency) / (max_latency - min_latency)
    return res

