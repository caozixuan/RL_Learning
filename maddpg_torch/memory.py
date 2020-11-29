import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 存储历史状态
class Memory(object):

    def __init__(self, size):
        self.size = size
        self.pointer = 0
        self.memory = []

    def clear(self):
        self.memory = []
        self.pointer = 0

    def add(self, s, a, r, s_, done, s_g=None, a_g=None):
        data = (s, a, r, s_, done, s_g, a_g)
        if self.pointer >= len(self.memory):
            self.memory.append(data)
        else:
            self.memory[self.pointer] = data
        self.pointer = (self.pointer + 1) % self.size

    def make_index(self, batch_size):
        return [random.randint(0, len(self.memory) - 1) for _ in range(batch_size)]

    # 将数据转化为pytorch tensor
    def encode(self, ids):
        s, a, r, s_, dones = [], [], [], [], []
        for i in ids:
            data = self.memory[i]
            si, ai, ri, s_i, done, s_g, a_g = data
            s.append(si)
            a.append(ai)
            r.append(ri)
            s_.append(s_i)
            dones.append(done)
        return torch.FloatTensor(s).to(device), torch.FloatTensor(a).to(device), torch.FloatTensor(r).to(
            device), torch.FloatTensor(
            s_).to(device), torch.FloatTensor(dones).to(device)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
        if batch_size > 0:
            ids = self.make_index(batch_size)
        else:
            ids = range(0, len(self.memory))
        return self.encode(ids)
