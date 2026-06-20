import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.integrate import odeint
from torch.distributions import Normal

v_min, v_max = -1e3, 1e3
thresh = 0.8
lens = 0.4
decay = 0.2
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

def mem_update(x, mem, spike):
    mem1 = mem * decay * (1. - spike) + x
    spike1 = act_fun(mem1)
    return mem1, spike1

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def _topk_slot_select(spikes, topk):
    """
    Drop slots per sample by lowest total spike count, gather the rest.

    spikes: [batch, wins, channel, num_slots]
    Returns: [batch, wins, channel, topk]
    """
    if topk is None or topk >= spikes.size(-1):
        return spikes
    spike_counts = spikes.sum(dim=(1, 2))                # [batch, num_slots]
    _, sorted_idx = torch.sort(spike_counts, dim=1, descending=True)
    top_idx = sorted_idx[:, :topk]                       # [batch, topk]
    batch_size, wins, channel, _ = spikes.size()
    gather_idx = top_idx.view(batch_size, 1, 1, topk).expand(-1, wins, channel, -1)
    return spikes.gather(3, gather_idx)

class LIF_1_2_2_neuron(nn.Module):
    """
    1+2+2 LIF topology (mem-mixing, LIF_hh style, 5 LIF slots).

    Slot layout: [batch, channel, 5] (or [T, batch, channel, 5])
        slot 0     -> 1 primary LIF (processes input)
        slots 1, 2 -> 2 stage-2 mixing LIFs, see OLD mem[:,:,0] via
                      lif_fc1 = nn.Linear(1, 2) (non-negative)
        slots 3, 4 -> 2 stage-3 mixing LIFs, see OLD (mem[:,:,1], mem[:,:,2])
                      via lif_fc2 = nn.Linear(2, 2) (non-negative)

    Args:
        topk: int or None. If < 5, dynamically drops the (5 - topk) slots
              with the lowest total spike count per sample. None keeps all 5.
    """
    def __init__(self, in_planes, out_planes, topk=5):
        super(LIF_1_2_2_neuron, self).__init__()
        self.fc1 = nn.Linear(in_planes, out_planes)
        self.lif_fc1 = nn.Linear(1, 2).to(device)
        self.lif_fc1.weight.data = abs(self.lif_fc1.weight.data)
        self.lif_fc2 = nn.Linear(2, 2).to(device)
        self.lif_fc2.weight.data = abs(self.lif_fc2.weight.data)
        self.channel = out_planes
        self.thresh = thresh
        self.topk = topk
        self.apply(weights_init_)

    def update_neuron(self, input, mem, spike):
        if input.dim() == 2:
            input_all = torch.zeros_like(mem)
            input_all[:, :, 0] = self.fc1(input)
            inner1 = self.lif_fc1(mem[:, :, 0:1])
            input_all[:, :, 1] = inner1[:, :, 0]
            input_all[:, :, 2] = inner1[:, :, 1]
            inner2 = self.lif_fc2(mem[:, :, 1:3])
            input_all[:, :, 3] = inner2[:, :, 0]
            input_all[:, :, 4] = inner2[:, :, 1]
        else:
            input_all = torch.zeros_like(mem)
            input_all[:, :, :, 0] = self.fc1(input)
            inner1 = self.lif_fc1(mem[:, :, :, 0:1])
            input_all[:, :, :, 1] = inner1[:, :, :, 0]
            input_all[:, :, :, 2] = inner1[:, :, :, 1]
            inner2 = self.lif_fc2(mem[:, :, :, 1:3])
            input_all[:, :, :, 3] = inner2[:, :, :, 0]
            input_all[:, :, :, 4] = inner2[:, :, :, 1]
        mem1 = torch.zeros_like(mem, device=device)
        spike_out = torch.zeros_like(spike, device=device)
        mem1, spike_out = mem_update(input_all, mem, spike)
        return mem1, spike_out

    def forward(self, input, wins=15):
        input = input.float().to(device)
        if input.dim() == 2:
            batch_size = input.size(0)
            mem = torch.zeros([batch_size, self.channel, 5]).to(device)
            spike = torch.zeros([batch_size, self.channel, 5]).to(device)
            spikes = torch.zeros([batch_size, wins, self.channel, 5]).to(device)
            for step in range(wins):
                mem, spike = self.update_neuron(input, mem, spike)
                spikes[:, step, ...] = spike
            spikes = _topk_slot_select(spikes, self.topk)
            spikes = spikes.contiguous().view(batch_size, wins, -1)
        else:
            batch_size = input.size(1)
            mem = torch.zeros([input.size(0), batch_size, self.channel, 5]).to(device)
            spike = torch.zeros([input.size(0), batch_size, self.channel, 5]).to(device)
            spikes = torch.zeros([input.size(0), batch_size, wins, self.channel, 5]).to(device)
            for step in range(wins):
                mem, spike = self.update_neuron(input, mem, spike)
                spikes[:, :, step, ...] = spike
            spikes = spikes.view(input.size(0), batch_size, wins, -1)
        return spikes
