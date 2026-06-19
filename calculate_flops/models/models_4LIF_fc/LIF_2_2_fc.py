# -*- coding: utf-8 -*-
"""
LIF_2_2 model: 2+2 LIF topology for multi-task learning (mem-mixing design,
mirroring LIF_hh).

Topology
--------
- Stage 1: 2 primary LIFs process the raw input image (slots 0, 1)
- Stage 2: 2 mixing LIFs (slots 2, 3) each receive a learned linear
  combination of the stage-1 LIFs' mems via lif_fc = nn.Linear(2, 2) with
  non-negative weights (matching LIF_hh's pattern for its single residual
  LIF)

Total: 4 LIFs, same count as LIF_HH (3+1) and LIF_1_3 (1+3).
Compared with 1+3, the stage-1 capacity is doubled (2 primaries) so the
stage-2 mixing has more source mems to integrate.

Memory / spike tensor layout: [batch, out_planes, 4]
    slot 0  -> stage-1 LIF (primary, branch a)
    slot 1  -> stage-1 LIF (primary, branch b)
    slot 2  -> stage-2 LIF (mixing, output 0 of lif_fc)
    slot 3  -> stage-2 LIF (mixing, output 1 of lif_fc)
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


class lif_2_2(nn.Module):
    """
    2+2 LIF sub-network (4 LIFs total, mem-mixing, LIF_hh style).

    Forward flow at each timestep:
        input -> fc_a, fc_b -> stage-1 LIFs (slots 0, 1) -> mem_a, mem_b
                                                              |
                              lif_fc(mem_a, mem_b)  (Linear(2, 2), abs weight)
                                                              |
                            +---------------+
                            |               |
                       stage-2 LIF    stage-2 LIF
                       (slot 2)       (slot 3)

    The 2 stage-2 LIFs do not see `input` directly; they only see the
    mems of the stage-1 LIFs, through a small non-negative mixing matrix.
    """
    def __init__(self, in_planes, out_planes):
        super(lif_2_2, self).__init__()
        # 2 primary LIFs
        self.fc_a = nn.Linear(in_planes, out_planes)
        self.fc_b = nn.Linear(in_planes, out_planes)
        # 1 mixing layer producing 2 input currents (one per stage-2 LIF)
        self.lif_fc = nn.Linear(2, 2)
        self.lif_fc.weight.data = abs(self.lif_fc.weight.data)

    def forward(self, input, mem, spike):
        # mem, spike: [batch, out_planes, 4]
        # All out-of-place to keep autograd happy.

        # ---- Stage 1: 2 primary LIFs from the raw input ----
        in_a = self.fc_a(input)
        in_b = self.fc_b(input)
        mem_a, spike_a = mem_update(in_a, mem[..., 0], spike[..., 0])
        mem_b, spike_b = mem_update(in_b, mem[..., 1], spike[..., 1])

        # ---- Stage 2: 2 mixing LIFs from (mem_a, mem_b) ----
        inner = self.lif_fc(mem[..., 0:2])  # [batch, out_planes, 2]
        mem_c, spike_c = mem_update(inner[..., 0], mem[..., 2], spike[..., 2])
        mem_d, spike_d = mem_update(inner[..., 1], mem[..., 3], spike[..., 3])

        mem_new = torch.stack([mem_a, mem_b, mem_c, mem_d], dim=-1)
        spike_new = torch.stack([spike_a, spike_b, spike_c, spike_d], dim=-1)
        return mem_new, spike_new


class SNN_Model_LIF_2_2(nn.Module):

    def __init__(self, n_tasks):
        super(SNN_Model_LIF_2_2, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

        self.fc_output = nn.Linear(cfg_fc[0] * 4, cfg_fc[1])
        self.lif_4 = lif_2_2(36 * 36 * 1, cfg_fc[0])

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
