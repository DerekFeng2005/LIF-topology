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

class LIF_1_3_neuron(nn.Module):
    """
    1+3 LIF topology (alternative to LIF_HH's 3+1):
        - Stage 1: 1 LIF processes the raw input (slot 0)
        - Stage 2: 3 parallel LIFs each receive the stage-1 spike
                    as their input current (slots 1, 2, 3)

    Total: 4 LIFs, same as LIF_HH, but with a strict two-stage
    information flow rather than 3-parallel + 1-residual.
    """
    def __init__(self, in_planes, out_planes):
        super(LIF_1_3_neuron, self).__init__()
        self.channel = out_planes
        self.fc_in = nn.Linear(in_planes, out_planes)
        self.fc_a = nn.Linear(out_planes, out_planes)
        self.fc_b = nn.Linear(out_planes, out_planes)
        self.fc_c = nn.Linear(out_planes, out_planes)

    def forward(self, input, wins=15):
        batch_size = input.size(0)
        dev = input.device

        # mem, spike: [batch, channel, 4]
        mem = torch.zeros([batch_size, self.channel, 4], device=dev)
        spike = torch.zeros([batch_size, self.channel, 4], device=dev)
        spikes = torch.zeros([batch_size, wins, self.channel, 4], device=dev)

        for step in range(wins):
            x = input[:, step, ...]

            # ---- Stage 1: 1 LIF from the raw input ----
            in0 = self.fc_in(x)
            mem0, spike0 = mem_update(in0, mem[:,:,0], spike[:,:,0])

            # ---- Stage 2: 3 LIFs from the stage-1 spike ----
            in_a = self.fc_a(spike0)
            in_b = self.fc_b(spike0)
            in_c = self.fc_c(spike0)
            mem_a, spike_a = mem_update(in_a, mem[:,:,1], spike[:,:,1])
            mem_b, spike_b = mem_update(in_b, mem[:,:,2], spike[:,:,2])
            mem_c, spike_c = mem_update(in_c, mem[:,:,3], spike[:,:,3])

            # Out-of-place stack to keep autograd happy
            mem = torch.stack([mem0, mem_a, mem_b, mem_c], dim=-1)
            spike = torch.stack([spike0, spike_a, spike_b, spike_c], dim=-1)

            spikes[:, step, ...] = spike

        spikes = spikes.view(batch_size, wins, -1)
        return spikes


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.lif_1_3_layer = LIF_1_3_neuron(num_inputs, hidden_dim)
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
        x = self.lif_1_3_layer(state)
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
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
