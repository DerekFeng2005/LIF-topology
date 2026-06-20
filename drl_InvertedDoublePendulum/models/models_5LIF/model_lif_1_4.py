import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

v_min, v_max = -1e3, 1e3
thresh = 0.8
lens = 0.4
decay = 0.2
device = torch.device("cpu")

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

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

def _topk_slot_select(spikes, topk):
    """spikes: [batch, wins, channel, 5]. Return topk slots by total spike count per sample."""
    if topk is None or topk >= spikes.size(-1):
        return spikes
    spike_counts = spikes.sum(dim=(1, 2))  # [batch, 5]
    sorted_idx = torch.argsort(spike_counts, dim=1, descending=True)
    top_idx = sorted_idx[:, :topk]  # [batch, topk]
    batch_size, wins, channel, _ = spikes.size()
    gather_idx = top_idx.view(batch_size, 1, 1, topk).expand(-1, wins, channel, -1)
    return spikes.gather(3, gather_idx)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class LIF_1_4_neuron(nn.Module):
    """
    1+4 LIF topology (mem-mixing, LIF_hh style).

    Slot layout: [batch, channel, 5]
        slot 0          -> 1 primary LIF (processes input)
        slots 1, 2, 3, 4 -> 4 mixing LIFs, each receives a learned non-negative
                            scalar of mem[:,:,0] via lif_fc = nn.Linear(1, 4)
    """
    def __init__(self, in_planes, out_planes, topk=5):
        super(LIF_1_4_neuron, self).__init__()
        self.fc1 = nn.Linear(in_planes, out_planes)
        self.lif_fc = nn.Linear(1, 4)
        self.lif_fc.weight.data = abs(self.lif_fc.weight.data)
        self.channel = out_planes
        self.topk = topk
        self.thresh = thresh
        self.apply(weights_init_)

    def update_neuron(self, input, mem, spike):
        input_all = torch.zeros_like(mem)
        input_all[:, :, 0] = self.fc1(input)
        inner = self.lif_fc(mem[:, :, 0:1])
        input_all[:, :, 1] = inner[:, :, 0]
        input_all[:, :, 2] = inner[:, :, 1]
        input_all[:, :, 3] = inner[:, :, 2]
        input_all[:, :, 4] = inner[:, :, 3]
        mem1 = torch.zeros_like(mem)
        spike_out = torch.zeros_like(spike)
        mem1, spike_out = mem_update(input_all, mem, spike)
        return mem1, spike_out

    def forward(self, input, wins=15):
        dev = input.device
        input = input.float()
        batch_size = input.size(0)
        mem = torch.zeros([batch_size, self.channel, 5], device=dev)
        spike = torch.zeros([batch_size, self.channel, 5], device=dev)
        spikes = torch.zeros([batch_size, wins, self.channel, 5], device=dev)
        for step in range(wins):
            mem, spike = self.update_neuron(input[:, step, ...], mem, spike)
            spikes[:, step, ...] = spike
        spikes = _topk_slot_select(spikes, self.topk)
        spikes = spikes.view(batch_size, wins, -1)
        return spikes


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, topk=5):
        super(GaussianPolicy, self).__init__()

        self.lif_1_4_layer1 = LIF_1_4_neuron(num_inputs, hidden_dim, topk=topk)
        self.lif_1_4_layer2 = LIF_1_4_neuron(topk*hidden_dim, hidden_dim, topk=topk)
        self.linear1_1 = nn.Linear(topk*hidden_dim, hidden_dim)
        self.linear1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2_1 = nn.Linear(topk*hidden_dim, hidden_dim)
        self.linear2_2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        input_tmp = []
        for i in range(5):
            input_tmp += [state[:,i,...]]*3
        state = torch.stack(input_tmp, dim=1)
        x = self.lif_1_4_layer1(state)
        x = torch.mean(x, dim=1)
        x1 = self.linear1_1(x)
        x1 = nn.ReLU()(x1)
        x1 = self.linear1_2(x1)
        x1 = nn.ReLU()(x1)
        mean = self.mean_linear(x1)
        x2 = self.linear2_1(x)
        x2 = nn.ReLU()(x2)
        x2 = self.linear2_2(x2)
        x2 = nn.ReLU()(x2)
        log_std = self.log_std_linear(x2)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        log_prob = log_prob.flatten(1)  # collapse wins dim -> [batch, num_actions]
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
