import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss

device = torch.device("cuda:0")
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


class lif_ring(nn.Module):
    """
    4 LIFs in a cross-timestep ring (0 -> 1 -> 2 -> 3 -> 0, with 1-step delay).

    Each LIF has its own conv layer for external input. The recurrent part
    uses the previous step's mem from the LIF preceding it in the ring,
    with 0 learnable recurrent weights.
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(lif_ring, self).__init__()
        # 4 separate external convs, in_planes -> out_planes
        self.conv_0 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                stride=stride, padding=padding).to(device)
        self.conv_1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                stride=stride, padding=padding).to(device)
        self.conv_2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                stride=stride, padding=padding).to(device)
        self.conv_3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                stride=stride, padding=padding).to(device)

    def forward(self, input, mem, spike, is_spike_input=True):
        """
        input:
            5D [batch, C, H, W, 4]  when is_spike_input=True
            4D [batch, C, H, W]      when is_spike_input=False
        mem, spike: 5D [batch, C', H', W', 4]
        """
        # ---- External input (4 separate convs) ----
        if is_spike_input:
            in_0_ext = (self.conv_0(input[:,:,:,:,0])
                        + self.conv_0(input[:,:,:,:,1])
                        + self.conv_0(input[:,:,:,:,2])
                        + self.conv_0(input[:,:,:,:,3]))
            in_1_ext = (self.conv_1(input[:,:,:,:,0])
                        + self.conv_1(input[:,:,:,:,1])
                        + self.conv_1(input[:,:,:,:,2])
                        + self.conv_1(input[:,:,:,:,3]))
            in_2_ext = (self.conv_2(input[:,:,:,:,0])
                        + self.conv_2(input[:,:,:,:,1])
                        + self.conv_2(input[:,:,:,:,2])
                        + self.conv_2(input[:,:,:,:,3]))
            in_3_ext = (self.conv_3(input[:,:,:,:,0])
                        + self.conv_3(input[:,:,:,:,1])
                        + self.conv_3(input[:,:,:,:,2])
                        + self.conv_3(input[:,:,:,:,3]))
        else:
            in_0_ext = self.conv_0(input)
            in_1_ext = self.conv_1(input)
            in_2_ext = self.conv_2(input)
            in_3_ext = self.conv_3(input)

        # ---- Ring recurrent input: previous step's mem from preceding LIF ----
        # LIF_0 <- slot 3, LIF_1 <- slot 0, LIF_2 <- slot 1, LIF_3 <- slot 2
        in_0 = in_0_ext + mem[:,:,:,:,3]
        in_1 = in_1_ext + mem[:,:,:,:,0]
        in_2 = in_2_ext + mem[:,:,:,:,1]
        in_3 = in_3_ext + mem[:,:,:,:,2]

        mem0, spike0 = mem_update(in_0, mem[:,:,:,:,0], spike[:,:,:,:,0])
        mem1, spike1 = mem_update(in_1, mem[:,:,:,:,1], spike[:,:,:,:,1])
        mem2, spike2 = mem_update(in_2, mem[:,:,:,:,2], spike[:,:,:,:,2])
        mem3, spike3 = mem_update(in_3, mem[:,:,:,:,3], spike[:,:,:,:,3])

        mem_new = torch.stack([mem0, mem1, mem2, mem3], dim=-1)
        spike_new = torch.stack([spike0, spike1, spike2, spike3], dim=-1)
        return mem_new, spike_new


class SCNN_Model_LIF_ring(nn.Module):
    def __init__(self, n_tasks):
        super(SCNN_Model_LIF_ring, self).__init__()
        self.n_tasks = n_tasks
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.lif_4_1 = lif_ring(in_planes, out_planes, kernel_size, stride, padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.lif_4_2 = lif_ring(in_planes, out_planes, kernel_size, stride, padding)
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
