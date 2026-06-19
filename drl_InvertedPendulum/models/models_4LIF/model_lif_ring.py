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

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class LIF_ring_neuron(nn.Module):
    """
    4 LIFs in a cross-timestep ring (0 -> 1 -> 2 -> 3 -> 0, with 1-step delay).
    Each LIF has its own m->n external projection. LIF_k receives its own
    external input plus the previous step's mem from LIF_{(k-1) mod 4}.
    Returns 4*channel features per timestep.
    """
    def __init__(self, in_planes, out_planes):
        super(LIF_ring_neuron, self).__init__()

        self.fc_0 = nn.Linear(in_planes, out_planes)
        self.fc_1 = nn.Linear(in_planes, out_planes)
        self.fc_2 = nn.Linear(in_planes, out_planes)
        self.fc_3 = nn.Linear(in_planes, out_planes)
        self.channel = out_planes
        self.thresh = thresh
        self.apply(weights_init_)

    def update_neuron(self, input, mem, spike):
        if input.dim() == 2:
            input_all = torch.zeros_like(mem)
            input_all[:, :, 0] = self.fc_0(input) + mem[:, :, 3]
            input_all[:, :, 1] = self.fc_1(input) + mem[:, :, 0]
            input_all[:, :, 2] = self.fc_2(input) + mem[:, :, 1]
            input_all[:, :, 3] = self.fc_3(input) + mem[:, :, 2]
        else:
            input_all = torch.zeros_like(mem)
            input_all[:, :, :, 0] = self.fc_0(input) + mem[:, :, :, 3]
            input_all[:, :, :, 1] = self.fc_1(input) + mem[:, :, :, 0]
            input_all[:, :, :, 2] = self.fc_2(input) + mem[:, :, :, 1]
            input_all[:, :, :, 3] = self.fc_3(input) + mem[:, :, :, 2]
        mem1 = torch.zeros_like(mem)
        spike_out = torch.zeros_like(spike)
        mem1, spike_out = mem_update(input_all, mem, spike)
        return mem1, spike_out

    def forward(self, input, wins=15):
        dev = input.device
        input = input.float()
        if input.dim() == 2:
            batch_size = input.size(0)
            mem = torch.zeros([batch_size, self.channel, 4], device=dev)
            spike = torch.zeros([batch_size, self.channel, 4], device=dev)
            spikes = torch.zeros([batch_size, wins, self.channel, 4], device=dev)
            for step in range(wins):
                mem, spike = self.update_neuron(input, mem, spike)
                spikes[:, step, ...] = spike
            spikes = spikes.view(batch_size, wins, -1)
        else:
            batch_size = input.size(0)
            mem = torch.zeros([batch_size, self.channel, 4], device=dev)
            spike = torch.zeros([batch_size, self.channel, 4], device=dev)
            spikes = torch.zeros([batch_size, wins, self.channel, 4], device=dev)
            for step in range(wins):
                mem, spike = self.update_neuron(input[:, step, ...], mem, spike)
                spikes[:, step, ...] = spike
            spikes = spikes.view(batch_size, wins, -1)
        return spikes


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.lif_ring_layer = LIF_ring_neuron(num_inputs, hidden_dim)
        self.linear1_1 = nn.Linear(4*hidden_dim, hidden_dim)
        self.linear1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2_1 = nn.Linear(4*hidden_dim, hidden_dim)
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
        x = self.lif_ring_layer(state)
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
