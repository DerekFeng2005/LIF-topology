import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss

device = torch.device("cpu")
thresh = 0.3
lens = 0.4
decay = 0.2

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

cfg_cnn = [(1, 10, 1, 1, 3),
           (10, 10, 1, 1, 3),]
cfg_kernel = [36, 18, 9]
cfg_fc = [512, 50]


class lif_1_3(nn.Module):
    """
    1+3 LIF topology (conv variant, mem-mixing, LIF_hh style).

    Forward flow at each timestep:
        input -> conv_in -> stage-1 LIF (slot 0) -> mem_0
                                                       |
                                  lif_fc(mem_0)  (Linear(1, 3), abs weight, per-channel)
                                                       |
                            +----------+--------------+
                            |          |              |
                       stage-2    stage-2        stage-2
                       LIF (1)    LIF (2)        LIF (3)

    The 3 stage-2 LIFs do not see `input` directly; they only see the
    stage-1 LIF's mem through a small non-negative per-channel mixing
    matrix.
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(lif_1_3, self).__init__()
        # 1 primary LIF (1 conv on the (summed) input)
        self.conv_in = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                 stride=stride, padding=padding).to(device)
        # 1 mixing layer producing 3 input currents (one per stage-2 LIF)
        # Linear (not Conv) because the mixing is per-channel only.
        self.lif_fc = nn.Linear(1, 3).to(device)
        self.lif_fc.weight.data = abs(self.lif_fc.weight.data)

    def forward(self, input, mem, spike, is_spike_input=True):
        """
        input:
            5D [batch, C, H, W, 4]  when is_spike_input=True
                (spikes from previous layer's 4 LIF slots)
            4D [batch, C, H, W]      when is_spike_input=False
                (raw conv input, e.g. image)
        mem, spike: 5D [batch, C', H', W', 4]
        """
        # ---- Stage 1: 1 primary LIF from the input ----
        if is_spike_input:
            in0 = (self.conv_in(input[:,:,:,:,0])
                   + self.conv_in(input[:,:,:,:,1])
                   + self.conv_in(input[:,:,:,:,2])
                   + self.conv_in(input[:,:,:,:,3]))
        else:
            in0 = self.conv_in(input)
        mem0, spike0 = mem_update(in0, mem[:,:,:,:,0], spike[:,:,:,:,0])

        # ---- Stage 2: 3 mixing LIFs from mem_0 ----
        # mem[:,:,:,:,0:1] is 5D [batch, C', H', W', 1]; lif_fc mixes the
        # last dim, producing 5D [batch, C', H', W', 3].
        inner = self.lif_fc(mem[:,:,:,:,0:1])
        mem_a, spike_a = mem_update(inner[:,:,:,:,0], mem[:,:,:,:,1], spike[:,:,:,:,1])
        mem_b, spike_b = mem_update(inner[:,:,:,:,1], mem[:,:,:,:,2], spike[:,:,:,:,2])
        mem_c, spike_c = mem_update(inner[:,:,:,:,2], mem[:,:,:,:,3], spike[:,:,:,:,3])

        mem_new = torch.stack([mem0, mem_a, mem_b, mem_c], dim=-1)
        spike_new = torch.stack([spike0, spike_a, spike_b, spike_c], dim=-1)
        return mem_new, spike_new


class SCNN_Model_LIF_1_3(nn.Module):
    def __init__(self, n_tasks):
        super(SCNN_Model_LIF_1_3, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.lif_4_1 = lif_1_3(in_planes, out_planes, kernel_size, stride, padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.lif_4_2 = lif_1_3(in_planes, out_planes, kernel_size, stride, padding)
        self.fc = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1] * 4,
                            cfg_fc[1]).to(device)

    def forward(self, input, time_window=10):
        batch_size = input.size(0)
        c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0],
                               cfg_kernel[0], 4, device=device)
        c1_mem = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0],
                             cfg_kernel[0], 4, device=device)
        c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1],
                               cfg_kernel[1], 4, device=device)
        c2_mem = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1],
                             cfg_kernel[1], 4, device=device)
        h1_sumspike = torch.zeros(batch_size,
                                  cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1] * 4,
                                  device=device)

        for step in range(time_window):
            c1_mem, c1_spike = self.lif_4_1(input, c1_mem, c1_spike, is_spike_input=False)
            x = torch.zeros(c1_spike.size(0), c1_spike.size(1),
                            c1_spike.size(2) // 2, c1_spike.size(3) // 2, 4,
                            device=device)
            x[:,:,:,:,0] = F.avg_pool2d(c1_spike[:,:,:,:,0], 2)
            x[:,:,:,:,1] = F.avg_pool2d(c1_spike[:,:,:,:,1], 2)
            x[:,:,:,:,2] = F.avg_pool2d(c1_spike[:,:,:,:,2], 2)
            x[:,:,:,:,3] = F.avg_pool2d(c1_spike[:,:,:,:,3], 2)
            c2_mem, c2_spike = self.lif_4_2(x, c2_mem, c2_spike, is_spike_input=True)
            x = torch.zeros(c2_spike.size(0), c2_spike.size(1),
                            c2_spike.size(2) // 2, c2_spike.size(3) // 2, 4,
                            device=device)
            x[:,:,:,:,0] = F.avg_pool2d(c2_spike[:,:,:,:,0], 2)
            x[:,:,:,:,1] = F.avg_pool2d(c2_spike[:,:,:,:,1], 2)
            x[:,:,:,:,2] = F.avg_pool2d(c2_spike[:,:,:,:,2], 2)
            x[:,:,:,:,3] = F.avg_pool2d(c2_spike[:,:,:,:,3], 2)

            x_flat = x.view(batch_size, -1)
            h1_sumspike = h1_sumspike + x_flat

        outs = self.fc(h1_sumspike / time_window)

        output = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            output.append(layer(outs))
        return torch.stack(output, dim=1)
