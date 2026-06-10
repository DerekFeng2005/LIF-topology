# -*- coding: utf-8 -*-
"""
LIF_1_2_1 model: 1+2+1 LIF topology for multi-task learning.

Topology
--------
- Stage 1: 1 LIF processes the raw input image (slot 0)
- Stage 2: 2 parallel LIFs each receive spike0 (slots 1, 2)
- Stage 3: 1 LIF receives (spike_a + spike_b) (slot 3)

Total: 4 LIFs, same count as the other 4-LIF variants.
Stage-1 has 1 LIF (128 spike bits, same as LIF_1_3); stage-3 has 1 LIF
(another bottleneck). Adds one extra layer of depth vs LIF_1_3.

Memory / spike tensor layout: [batch, out_planes, 4]
    slot 0  -> stage-1 LIF
    slot 1  -> stage-2 LIF (branch a)
    slot 2  -> stage-2 LIF (branch b)
    slot 3  -> stage-3 LIF
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


class lif_1_2_1(nn.Module):
    """
    1+2+1 LIF sub-network (4 LIFs total, 3 stages).

    Forward flow at each timestep:
        input -> fc_in -> stage-1 LIF (slot 0) -> spike0
                                                  |
                            +---------------------+
                            |                     |
                        fc_a, fc_b             (sum)
                            |                     |
                       stage-2 LIFs          stage-2
                       (slots 1, 2)          LIF (slot 1, 2)
                                                  |
                                                (spike_a + spike_b)
                                                  |
                                                fc_c
                                                  |
                                              stage-3 LIF (slot 3)
    """
    def __init__(self, in_planes, out_planes):
        super(lif_1_2_1, self).__init__()
        # Stage 1: 1 LIF, projects raw input -> hidden
        self.fc_in = nn.Linear(in_planes, out_planes)
        # Stage 2: 2 parallel LIFs from spike0
        self.fc_a = nn.Linear(out_planes, out_planes)
        self.fc_b = nn.Linear(out_planes, out_planes)
        # Stage 3: 1 LIF from (spike_a + spike_b)
        self.fc_c = nn.Linear(out_planes, out_planes)

    def forward(self, input, mem, spike):
        # mem, spike: [batch, out_planes, 4]
        # All out-of-place to keep autograd happy.

        # ---- Stage 1: 1 LIF from the raw input ----
        in0 = self.fc_in(input)
        mem0, spike0 = mem_update(in0, mem[..., 0], spike[..., 0])

        # ---- Stage 2: 2 LIFs from spike0 ----
        in_a = self.fc_a(spike0)
        in_b = self.fc_b(spike0)
        mem_a, spike_a = mem_update(in_a, mem[..., 1], spike[..., 1])
        mem_b, spike_b = mem_update(in_b, mem[..., 2], spike[..., 2])

        # ---- Stage 3: 1 LIF from (spike_a + spike_b) ----
        spike_sum = spike_a + spike_b
        in_c = self.fc_c(spike_sum)
        mem_c, spike_c = mem_update(in_c, mem[..., 3], spike[..., 3])

        mem_new = torch.stack([mem0, mem_a, mem_b, mem_c], dim=-1)
        spike_new = torch.stack([spike0, spike_a, spike_b, spike_c], dim=-1)

        return mem_new, spike_new


class SNN_Model_LIF_1_2_1(nn.Module):

    def __init__(self, n_tasks):
        super(SNN_Model_LIF_1_2_1, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

        self.fc_output = nn.Linear(cfg_fc[0] * 4, cfg_fc[1])
        self.lif_4 = lif_1_2_1(36 * 36 * 1, cfg_fc[0])

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
