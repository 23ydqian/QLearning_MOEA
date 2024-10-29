import numpy as np

'''
求解均值向量
'''

class Mean_vector:
    # 对m维空间，目标方向个数H
    def __init__(self, H=4, m=2, path='out.csv'):
        self.H = H
        self.m = m
        self.path = path
        self.stepsize = 1 / H

    def perm(self, sequence):
        # ！！！ 序列全排列，且无重复
        l = sequence
        if (len(l) <= 1):
            return [l]
        r = []
        for i in range(len(l)):
            if i != 0 and sequence[i - 1] == sequence[i]:       # 如果当前元素和前一个元素相同，那么就跳过这个元素。这是为了避免生成重复的排列。
                continue
            else:
                s = l[:i] + l[i + 1:]   # 原序列去掉第i个元素后
                p = self.perm(s)
                for x in p:
                    r.append(l[i:i + 1] + x)
        return r

    def get_mean_vectors(self):
        H = self.H
        m = self.m
        sequence = []
        # 在sequence中添加H个0和m-1个1
        for ii in range(H):
            sequence.append(0)
        for jj in range(m - 1):
            sequence.append(1)

        # 权重向量
        ws = []
        # 得到全排列
        pe_seq = self.perm(sequence)
        # 对每个排列进行遍历
        for sq in pe_seq:
            s = -1
            weight = []
            for i in range(len(sq)):    # 遍历排列中的每个元素
                if sq[i] == 1:
                    w = i - s
                    w = (w - 1) / H
                    s = i
                    weight.append(w)
            nw = H + m - 1 - s
            nw = (nw - 1) / H
            weight.append(nw)
            if weight not in ws:
                ws.append(weight)
        return ws

    def save_mv_to_file(self, mv):
        f = np.array(mv, dtype=np.float64)
        np.savetxt(fname=self.path, X=f)

    def generate(self):
        m_v = self.get_mean_vectors()
        self.save_mv_to_file(m_v)


# mv = Mean_vector(10, 3, 'test.csv')
# mv.generate()
