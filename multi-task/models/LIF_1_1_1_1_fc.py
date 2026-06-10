# -*- coding: utf-8 -*-
"""
LIF_1_1_1_1 model: 4 LIFs in serial for multi-task learning.

Topology
--------
- Stage 1: 1 LIF processes the raw input image (slot 0)
- Stage 2: 1 LIF receives spike0 (slot 1)
- Stage 3: 1 LIF receives spike1 (slot 2)
- Stage 4: 1 LIF receives spike2 (slot 3)

Total: 4 LIFs, no branching. Maximum depth (4) among the 4-LIF
variants; each stage's input is a single 128-bit binary spike vector
from the previous stage. Tests whether depth alone, with no width
advantage, helps over a single wide LIF layer (4xLIF).

Memory / spike tensor layout: [batch, out_planes, 4]
    slot 0  -> stage-1 LIF
    slot 1  -> stage-2 LIF
    slot 2  -> stage-3 LIF
    slot 3  -> stage-4 LIF
"""
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

v_min, v_max = -1e3, 1e3
thresh = 2
lens = 0.4
decay = 0.5
device = torch.device("cuda:0")


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


class lif_1_1_1_1(nn.Module):
    """
    4-stage serial LIF sub-network (4 LIFs total, no branching).

    Forward flow at each timestep:
        input -> fc_0 -> stage-1 LIF (slot 0) -> spike0
                                                  |
                                                fc_1
                                                  |
                                              stage-2 LIF (slot 1) -> spike1
                                                                          |
                                                                        fc_2
                                                                          |
                                                                      stage-3 LIF (slot 2) -> spike2
                                                                                                  |
                                                                                                fc_3
                                                                                                  |
                                                                                            stage-4 LIF (slot 3)
    """
    def __init__(self, in_planes, out_planes):
        super(lif_1_1_1_1, self).__init__()
        # 4 LIFs in serial, each transforms (spike from previous stage)
        self.fc_0 = nn.Linear(in_planes, out_planes)
        self.fc_1 = nn.Linear(out_planes, out_planes)
        self.fc_2 = nn.Linear(out_planes, out_planes)
        self.fc_3 = nn.Linear(out_planes, out_planes)

    def forward(self, input, mem, spike):
        # mem, spike: [batch, out_planes, 4]
        # All out-of-place to keep autograd happy.

        # ---- Stage 1: 1 LIF from raw input ----
        in0 = self.fc_0(input)
        mem0, spike0 = mem_update(in0, mem[..., 0], spike[..., 0])

        # ---- Stage 2: 1 LIF from spike0 ----
        in1 = self.fc_1(spike0)
        mem1, spike1 = mem_update(in1, mem[..., 1], spike[..., 1])

        # ---- Stage 3: 1 LIF from spike1 ----
        in2 = self.fc_2(spike1)
        mem2, spike2 = mem_update(in2, mem[..., 2], spike[..., 2])

        # ---- Stage 4: 1 LIF from spike2 ----
        in3 = self.fc_3(spike2)
        mem3, spike3 = mem_update(in3, mem[..., 3], spike[..., 3])

        mem_new = torch.stack([mem0, mem1, mem2, mem3], dim=-1)
        spike_new = torch.stack([spike0, spike1, spike2, spike3], dim=-1)

        return mem_new, spike_new


class SNN_Model_LIF_1_1_1_1(nn.Module):

    def __init__(self, n_tasks):
        super(SNN_Model_LIF_1_1_1_1, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

        self.fc_output = nn.Linear(cfg_fc[0] * 4, cfg_fc[1])
        self.lif_4 = lif_1_1_1_1(36 * 36 * 1, cfg_fc[0])

    def forward(self, input, win=15):
        batch_size = input.size(0)
        h1_mem = torch.zeros(batch_size, cfg_fc[0], 4, device=device)
        h1_spike = torch.zeros(batch_size, cfg_fc[0], 4, device=device)
        h1_sumspike = torch.zeros(batch_size, cfg_fc[0], 4, device=device)
        for step in range(win):
            x = input.view(batch_size, -1)
            h1_mem, h1_spike = self.lif_4(x, h1_mem, h1_spike)
            h1_sumspike = h1_sumspike + h1_spike

        x = h1_sumspike.view(batch_size, -1)
        outs = self.fc_output(x / win)

        output = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            output.append(layer(outs))
        return torch.stack(output, dim=1)


def mem_update(x, mem, spike):
    """Standard LIF update with reset-after-fire."""
    mem = mem * decay * (1 - spike) + x
    spike1 = act_fun(mem)
    return mem, spike1
