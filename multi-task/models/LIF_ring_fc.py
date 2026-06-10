# -*- coding: utf-8 -*-
"""
LIF_ring model: 4 LIFs in a cross-timestep ring for multi-task learning.

Topology
--------
- 4 LIFs in parallel within each timestep (DAG within timestep)
- Across timesteps, each LIF_k receives LIF_{(k-1) mod 4}'s previous-step
  mem as part of its current input. The unfolded computation graph over
  multiple timesteps is a ring (cycle): 0 -> 1 -> 2 -> 3 -> 0 -> ...

This is the standard SRNN ring pattern: acyclic within a single step,
but cyclic when the time axis is unfolded. No fixed-point iteration
needed; the previous step's mem is read directly from the state tensor.

Slot layout: [batch, out_planes, 4]
    slot 0: LIF_0
    slot 1: LIF_1
    slot 2: LIF_2
    slot 3: LIF_3

Ring routing (each LIF_k reads the previous-step mem from slot (k-1) mod 4):
    LIF_0 <- slot 3
    LIF_1 <- slot 0
    LIF_2 <- slot 1
    LIF_3 <- slot 2

Each LIF has its own m->n external projection (4 fcs total), so the
parameter count matches 4xLIF exactly: 4*m*n with 0 recurrent weights.
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


class lif_ring(nn.Module):
    """
    4 LIFs in a cross-timestep ring (0 -> 1 -> 2 -> 3 -> 0, with 1-step delay).

    Forward at each timestep t (with state (mem_{t-1}, spike_{t-1}) from t-1):
        in_0 = fc_0(x_t) + mem_{t-1, slot 3}
        in_1 = fc_1(x_t) + mem_{t-1, slot 0}
        in_2 = fc_2(x_t) + mem_{t-1, slot 1}
        in_3 = fc_3(x_t) + mem_{t-1, slot 2}
        LIF_k updates (mem, spike) in parallel from in_k
        Store new mem into state slots for the next step
    """
    def __init__(self, in_planes, out_planes):
        super(lif_ring, self).__init__()
        # 4 separate external projections, m -> n
        self.fc_0 = nn.Linear(in_planes, out_planes)
        self.fc_1 = nn.Linear(in_planes, out_planes)
        self.fc_2 = nn.Linear(in_planes, out_planes)
        self.fc_3 = nn.Linear(in_planes, out_planes)

    def forward(self, input, mem, spike):
        # mem, spike: [batch, out_planes, 4], from previous step (or zeros for t=0)
        # All out-of-place to keep autograd happy.

        # Each LIF: own external projection + previous step's mem from
        # the LIF preceding it in the ring.
        in_0 = self.fc_0(input) + mem[..., 3]
        in_1 = self.fc_1(input) + mem[..., 0]
        in_2 = self.fc_2(input) + mem[..., 1]
        in_3 = self.fc_3(input) + mem[..., 2]

        mem0, spike0 = mem_update(in_0, mem[..., 0], spike[..., 0])
        mem1, spike1 = mem_update(in_1, mem[..., 1], spike[..., 1])
        mem2, spike2 = mem_update(in_2, mem[..., 2], spike[..., 2])
        mem3, spike3 = mem_update(in_3, mem[..., 3], spike[..., 3])

        mem_new = torch.stack([mem0, mem1, mem2, mem3], dim=-1)
        spike_new = torch.stack([spike0, spike1, spike2, spike3], dim=-1)
        return mem_new, spike_new


class SNN_Model_LIF_ring(nn.Module):

    def __init__(self, n_tasks):
        super(SNN_Model_LIF_ring, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

        self.fc_output = nn.Linear(cfg_fc[0] * 4, cfg_fc[1])
        self.lif_4 = lif_ring(36 * 36 * 1, cfg_fc[0])

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
