
class FakeSynthesis:

    def __init__(self, entire_ds,):
        self.entire_ds = entire_ds

    def synthesise_configuration(self, config):
        c = config
        result = None
        for i in range(0, len(self.entire_ds)):
            if self.entire_ds[i].configuration == c:
                result = (float(self.entire_ds[i].latency), float(self.entire_ds[i].area), float(self.entire_ds[i].execute_time))
                break
        return result


