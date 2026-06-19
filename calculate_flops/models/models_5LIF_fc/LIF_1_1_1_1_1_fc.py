# -*- coding: utf-8 -*-
"""
LIF_1_1_1_1_1 model: 5-stage chain of 1 LIF per stage, total 5 LIFs.

Topology
--------
- Stage 1: 1 LIF processes the raw input image (slot 0)
- Stage k (k=2..5): 1 LIF (slot k-1) sees OLD mem[:,:,k-2] via
  lif_fc_{k-1} = nn.Linear(1, 1) (non-negative scalar)

Total: 5 LIFs. Parameter count: ~mn.
"""
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

v_min, v_max = -1e3, 1e3
thresh = 2
lens = 0.4
decay = 0.5
device = torch.device("cpu")


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()


act_fun = ActFun.apply

cfg_fc = [512, 50]


class lif_1_1_1_1_1(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(lif_1_1_1_1_1, self).__init__()
        self.fc_in = nn.Linear(in_planes, out_planes)
        self.lif_fc1 = nn.Linear(1, 1)
        self.lif_fc1.weight.data = abs(self.lif_fc1.weight.data)
        self.lif_fc2 = nn.Linear(1, 1)
        self.lif_fc2.weight.data = abs(self.lif_fc2.weight.data)
        self.lif_fc3 = nn.Linear(1, 1)
        self.lif_fc3.weight.data = abs(self.lif_fc3.weight.data)
        self.lif_fc4 = nn.Linear(1, 1)
        self.lif_fc4.weight.data = abs(self.lif_fc4.weight.data)

    def forward(self, input, mem, spike):
        in0 = self.fc_in(input)
        mem0, spike0 = mem_update(in0, mem[..., 0], spike[..., 0])

        # 4 sequential single-mem mixings
        inner1 = self.lif_fc1(mem[..., 0:1]).squeeze(-1)
        mem1, spike1 = mem_update(inner1, mem[..., 1], spike[..., 1])

        inner2 = self.lif_fc2(mem[..., 1:2]).squeeze(-1)
        mem2, spike2 = mem_update(inner2, mem[..., 2], spike[..., 2])

        inner3 = self.lif_fc3(mem[..., 2:3]).squeeze(-1)
        mem3, spike3 = mem_update(inner3, mem[..., 3], spike[..., 3])

        inner4 = self.lif_fc4(mem[..., 3:4]).squeeze(-1)
        mem4, spike4 = mem_update(inner4, mem[..., 4], spike[..., 4])

        mem_new = torch.stack([mem0, mem1, mem2, mem3, mem4], dim=-1)
        spike_new = torch.stack([spike0, spike1, spike2, spike3, spike4], dim=-1)
        return mem_new, spike_new


class SNN_Model_LIF_1_1_1_1_1(nn.Module):

    def __init__(self, n_tasks):
        super(SNN_Model_LIF_1_1_1_1_1, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

        self.fc_output = nn.Linear(cfg_fc[0] * 5, cfg_fc[1])
        self.lif_5 = lif_1_1_1_1_1(36 * 36 * 1, cfg_fc[0])

    def forward(self, input, win=15):
        batch_size = input.size(0)
        h1_mem = torch.zeros(batch_size, cfg_fc[0], 5, device=device)
        h1_spike = torch.zeros(batch_size, cfg_fc[0], 5, device=device)
        h1_sumspike = torch.zeros(batch_size, cfg_fc[0], 5, device=device)
        for step in range(win):
            x = input.view(batch_size, -1)
            h1_mem, h1_spike = self.lif_5(x, h1_mem, h1_spike)
            h1_sumspike = h1_sumspike + h1_spike

        x = h1_sumspike.view(batch_size, -1)
        outs = self.fc_output(x / win)

        output = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            output.append(layer(outs))
        return torch.stack(output, dim=1)


def mem_update(x, mem, spike):
    mem = mem * decay * (1 - spike) + x
    spike1 = act_fun(mem)
    return mem, spike1
