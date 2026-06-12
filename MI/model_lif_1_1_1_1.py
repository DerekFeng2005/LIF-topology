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

class LIF_1_1_1_1_neuron(nn.Module):
    """
    1+1+1+1 LIF topology (DAG chain, mem-mixing, LIF_hh style).

    Slot layout: [batch, channel, 4] (or [T, batch, channel, 4])
        slot 0 -> 1 primary LIF (processes input)
        slot 1 -> 1 stage-2 mixing LIF, sees OLD mem[:,:,0] via
                  lif_fc1 = nn.Linear(1, 1) (non-negative scalar)
        slot 2 -> 1 stage-3 mixing LIF, sees OLD mem[:,:,1] via
                  lif_fc2 = nn.Linear(1, 1) (non-negative scalar)
        slot 3 -> 1 stage-4 mixing LIF, sees OLD mem[:,:,2] via
                  lif_fc3 = nn.Linear(1, 1) (non-negative scalar)
    """
    def __init__(self, in_planes, out_planes):
        super(LIF_1_1_1_1_neuron, self).__init__()
        self.fc1 = nn.Linear(in_planes, out_planes)
        self.lif_fc1 = nn.Linear(1, 1).to(device)
        self.lif_fc1.weight.data = abs(self.lif_fc1.weight.data)
        self.lif_fc2 = nn.Linear(1, 1).to(device)
        self.lif_fc2.weight.data = abs(self.lif_fc2.weight.data)
        self.lif_fc3 = nn.Linear(1, 1).to(device)
        self.lif_fc3.weight.data = abs(self.lif_fc3.weight.data)
        self.channel = out_planes
        self.thresh = thresh
        self.apply(weights_init_)

    def update_neuron(self, input, mem, spike):
        if input.dim() == 2:
            input_all = torch.zeros_like(mem)
            input_all[:, :, 0] = self.fc1(input)
            input_all[:, :, 1] = self.lif_fc1(mem[:, :, 0:1]).squeeze(-1)
            input_all[:, :, 2] = self.lif_fc2(mem[:, :, 1:2]).squeeze(-1)
            input_all[:, :, 3] = self.lif_fc3(mem[:, :, 2:3]).squeeze(-1)
        else:
            input_all = torch.zeros_like(mem)
            input_all[:, :, :, 0] = self.fc1(input)
            input_all[:, :, :, 1] = self.lif_fc1(mem[:, :, :, 0:1]).squeeze(-1)
            input_all[:, :, :, 2] = self.lif_fc2(mem[:, :, :, 1:2]).squeeze(-1)
            input_all[:, :, :, 3] = self.lif_fc3(mem[:, :, :, 2:3]).squeeze(-1)
        mem1 = torch.zeros_like(mem, device=device)
        spike_out = torch.zeros_like(spike, device=device)
        mem1, spike_out = mem_update(input_all, mem, spike)
        return mem1, spike_out

    def forward(self, input, wins=15):
        input = input.float().to(device)
        if input.dim() == 2:
            batch_size = input.size(0)
            mem = torch.zeros([batch_size, self.channel, 4]).to(device)
            spike = torch.zeros([batch_size, self.channel, 4]).to(device)
            spikes = torch.zeros([batch_size, wins, self.channel, 4]).to(device)
            for step in range(wins):
                mem, spike = self.update_neuron(input, mem, spike)
                spikes[:, step, ...] = spike
            spikes = spikes.view(batch_size, wins, -1)
        else:
            batch_size = input.size(1)
            mem = torch.zeros([input.size(0), batch_size, self.channel, 4]).to(device)
            spike = torch.zeros([input.size(0), batch_size, self.channel, 4]).to(device)
            spikes = torch.zeros([input.size(0), batch_size, wins, self.channel, 4]).to(device)
            for step in range(wins):
                mem, spike = self.update_neuron(input, mem, spike)
                spikes[:, :, step, ...] = spike
            spikes = spikes.view(input.size(0), batch_size, wins, -1)
        return spikes
