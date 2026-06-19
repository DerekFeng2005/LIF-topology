# -*- coding: utf-8 -*-
"""
LIF_1_4 model: 1+4 LIF topology for multi-task learning (mem-mixing design,
5 LIF slots).

Topology
--------
- Stage 1: 1 LIF processes the raw input image (slot 0)
- Stage 2: 4 LIFs (slots 1, 2, 3, 4) each receive a learned non-negative
  scalar of the stage-1 LIF's mem via lif_fc = nn.Linear(1, 4)

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


class lif_1_4(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(lif_1_4, self).__init__()
        self.fc_in = nn.Linear(in_planes, out_planes)
        self.lif_fc = nn.Linear(1, 4)
        self.lif_fc.weight.data = abs(self.lif_fc.weight.data)

    def forward(self, input, mem, spike):
        in0 = self.fc_in(input)
        mem0, spike0 = mem_update(in0, mem[..., 0], spike[..., 0])

        # 4 mixing LIFs from mem0
        inner = self.lif_fc(mem[..., 0:1])
        mem_a, spike_a = mem_update(inner[..., 0], mem[..., 1], spike[..., 1])
        mem_b, spike_b = mem_update(inner[..., 1], mem[..., 2], spike[..., 2])
        mem_c, spike_c = mem_update(inner[..., 2], mem[..., 3], spike[..., 3])
        mem_d, spike_d = mem_update(inner[..., 3], mem[..., 4], spike[..., 4])

        mem_new = torch.stack([mem0, mem_a, mem_b, mem_c, mem_d], dim=-1)
        spike_new = torch.stack([spike0, spike_a, spike_b, spike_c, spike_d], dim=-1)
        return mem_new, spike_new


class SNN_Model_LIF_1_4(nn.Module):

    def __init__(self, n_tasks):
        super(SNN_Model_LIF_1_4, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

        self.fc_output = nn.Linear(cfg_fc[0] * 5, cfg_fc[1])
        self.lif_5 = lif_1_4(36 * 36 * 1, cfg_fc[0])

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
