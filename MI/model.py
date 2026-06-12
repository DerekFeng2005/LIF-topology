import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from utils import cuda

import time
from numbers import Number

from model_lif import LIF_neuron
from model_hh import HH_neuron
from model_lif_hh import LIF_hh_neuron
from model_lif_1_3 import LIF_1_3_neuron
from model_lif_2_2 import LIF_2_2_neuron
from model_lif_1_2_1 import LIF_1_2_1_neuron
from model_lif_1_1_1_1 import LIF_1_1_1_1_neuron
from model_lif_ring import LIF_ring_neuron

class ToyNet(nn.Module):

    def __init__(self, K=256, model="LIF"):
        super(ToyNet, self).__init__()
        self.K = K
        # Normalize so `--model LIF_hh` and `--model LIF_HH` both work.
        self.model = model.upper()
        if self.model == "LIF":
            self.neuron = LIF_neuron(1296,512)
        elif self.model == "HH":
            self.neuron = HH_neuron(1296,512)
        elif self.model == "LIF_HH":
            self.neuron = LIF_hh_neuron(1296,512)
        elif self.model == "LIF_1_3":
            self.neuron = LIF_1_3_neuron(1296,512)
        elif self.model == "LIF_2_2":
            self.neuron = LIF_2_2_neuron(1296,512)
        elif self.model == "LIF_1_2_1":
            self.neuron = LIF_1_2_1_neuron(1296,512)
        elif self.model == "LIF_1_1_1_1":
            self.neuron = LIF_1_1_1_1_neuron(1296,512)
        elif self.model == "LIF_RING":
            self.neuron = LIF_ring_neuron(1296,512)
        else:
            raise ValueError(
                "Unknown model: {!r}. Choose from: LIF, HH, LIF_HH, "
                "LIF_1_3, LIF_2_2, LIF_1_2_1, LIF_1_1_1_1, LIF_RING".format(model))

        if self.model in ("LIF", "HH"):
            self.encode = nn.Sequential(
                self.neuron,
                nn.Linear(512, 2*self.K))
        else:
            self.encode = nn.Sequential(
                self.neuron,
                nn.Linear(512*4, 2*self.K))
        
        for i in range(2):
            setattr(self, 'task_{}'.format(i), nn.Linear(self.K, 10))

    def forward(self, x, num_sample=1):
        if x.dim() > 2 : x = x.view(x.size(0),-1)

        statistics = torch.mean(self.encode(x),dim=1)
        mu = statistics[:,:self.K]
        std = F.softplus(statistics[:,self.K:]-5,beta=1)

        encoding = self.reparametrize_n(mu,std,num_sample)
        output = []
        for i in range(2):
            layer = getattr(self, 'task_{}'.format(i))
            output.append(layer(encoding))
        logit = torch.stack(output, dim=1)


        if num_sample == 1 : pass
        elif num_sample > 1 : logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()
