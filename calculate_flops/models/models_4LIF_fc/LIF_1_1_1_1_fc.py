# -*- coding: utf-8 -*-
"""
LIF_1_1_1_1 model: 4 LIFs in serial for multi-task learning (mem-mixing
design, mirroring LIF_hh).

Topology
--------
- Stage 1: 1 primary LIF processes the raw input image (slot 0)
- Stage 2: 1 mixing LIF (slot 1) receives a learned scalar of the stage-1
  LIF's mem via lif_fc1 = nn.Linear(1, 1) with non-negative weight
- Stage 3: 1 mixing LIF (slot 2) receives a learned scalar of the stage-2
  LIF's mem via lif_fc2 = nn.Linear(1, 1) with non-negative weight
- Stage 4: 1 mixing LIF (slot 3) receives a learned scalar of the stage-3
  LIF's mem via lif_fc3 = nn.Linear(1, 1) with non-negative weight

Total: 4 LIFs, no branching, no parallelism. Maximum depth (4) among the
4-LIF variants. Each stage's input is a single non-negative scalar multiple
of the previous stage's mem. Tests whether depth alone helps over a single
wide LIF layer (4xLIF) when the inter-stage channel is severely limited.

Memory / spike tensor layout: [batch, out_planes, 4]
    slot 0  -> stage-1 LIF (primary)
    slot 1  -> stage-2 LIF (mixing)
    slot 2  -> stage-3 LIF (mixing)
    slot 3  -> stage-4 LIF (mixing)
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


class lif_1_1_1_1(nn.Module):
    """
    4-stage serial LIF sub-network (4 LIFs total, no branching, mem-mixing,
    LIF_hh style).

    Forward flow at each timestep:
        input -> fc_0 -> stage-1 LIF (slot 0) -> mem_0
                                                     |
                                  lif_fc1(mem_0)     (Linear(1, 1), abs weight)
                                                     |
                                                stage-2 LIF (slot 1) -> mem_1
                                                                         |
                                                  lif_fc2(mem_1)        (Linear(1, 1), abs weight)
                                                                         |
                                                                    stage-3 LIF (slot 2) -> mem_2
                                                                                             |
                                                                      lif_fc3(mem_2)      (Linear(1, 1), abs weight)
                                                                                             |
                                                                                        stage-4 LIF (slot 3)

    Only the stage-1 LIF sees `input`; the 3 downstream LIFs only see the
    previous LIF's mem through a 1x1 (scalar) non-negative mixing.
    """
    def __init__(self, in_planes, out_planes):
        super(lif_1_1_1_1, self).__init__()
        # 1 primary LIF
        self.fc_0 = nn.Linear(in_planes, out_planes)
        # 3 scalar mixing layers (1->1 each, non-negative)
        self.lif_fc1 = nn.Linear(1, 1)
        self.lif_fc1.weight.data = abs(self.lif_fc1.weight.data)
        self.lif_fc2 = nn.Linear(1, 1)
        self.lif_fc2.weight.data = abs(self.lif_fc2.weight.data)
        self.lif_fc3 = nn.Linear(1, 1)
        self.lif_fc3.weight.data = abs(self.lif_fc3.weight.data)

    def forward(self, input, mem, spike):
        # mem, spike: [batch, out_planes, 4]
        # All out-of-place to keep autograd happy.

        # ---- Stage 1: 1 primary LIF from the raw input ----
        in0 = self.fc_0(input)
        mem0, spike0 = mem_update(in0, mem[..., 0], spike[..., 0])

        # ---- Stage 2: 1 mixing LIF from mem_0 ----
        inner1 = self.lif_fc1(mem[..., 0:1])[..., 0]  # [batch, out_planes]
        mem1, spike1 = mem_update(inner1, mem[..., 1], spike[..., 1])

        # ---- Stage 3: 1 mixing LIF from mem_1 ----
        inner2 = self.lif_fc2(mem[..., 1:2])[..., 0]
        mem2, spike2 = mem_update(inner2, mem[..., 2], spike[..., 2])

        # ---- Stage 4: 1 mixing LIF from mem_2 ----
        inner3 = self.lif_fc3(mem[..., 2:3])[..., 0]
        mem3, spike3 = mem_update(inner3, mem[..., 3], spike[..., 3])

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
