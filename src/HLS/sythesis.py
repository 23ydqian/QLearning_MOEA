import subprocess
import xml.etree.ElementTree

class FakeSynthesis:

    def __init__(self, entire_ds):
        self.entire_ds = entire_ds

    def synthesise_configuration(self, config):
        # config = list(config)
        result = None
        # 遍历整个设计空间
        for i in range(0, len(self.entire_ds)):
            # 第i个configuration如果与c
            if self.entire_ds[i].configuration == config:
                # 得到这个configuration对应的latency和area
                result = (self.entire_ds[i].latency, self.entire_ds[i].area,self.entire_ds[i].time)
                self.entire_ds[i].isSynthesis = 1
                break
        return result