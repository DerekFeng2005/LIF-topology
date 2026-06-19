"""OpsTracker: count Linear/Conv2d ops during a forward pass.

Wraps the `forward` method of each nn.Linear / nn.Conv2d module in the
given model with a counting version. The original class is preserved
(isinstance checks still pass), only the `forward` attribute is replaced.

Per-call accounting:
  nn.Linear(in, out):
    mul_op += in * out
    add_op += (in - 1) * out
  nn.Conv2d(Cin, Cout, k_h, k_w, ...):
    H', W' = output spatial size
    mul_op += Cin * Cout * k_h * k_w * H' * W'
    add_op += (Cin * k_h * k_w - 1) * Cout * H' * W'

Per-step neuron-internal ops (LIF mem/spike updates) are NOT counted
here; they are added separately based on the model's slot count and win
in the calling code (see spike_op_per_step helper).
"""
import torch
import torch.nn as nn


class OpsTracker:
    def __init__(self, model):
        self.model = model
        self._patches = []  # list of (module, original_forward)

    def wrap(self):
        for _, m in self.model.named_modules():
            if isinstance(m, nn.Linear) and not hasattr(m, "_ops_wrapped"):
                self._wrap_linear(m)
            elif isinstance(m, nn.Conv2d) and not hasattr(m, "_ops_wrapped"):
                self._wrap_conv2d(m)
        # mark all modules (so we don't double-wrap when called twice)
        for _, m in self.model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                m._ops_wrapped = True

    def _wrap_linear(self, m):
        orig = m.forward
        in_f = m.in_features
        out_f = m.out_features

        def counting_forward(x):
            self.mul_op += in_f * out_f
            self.add_op += (in_f - 1) * out_f
            return orig(x)

        m.forward = counting_forward
        self._patches.append((m, orig))

    def _wrap_conv2d(self, m):
        orig = m.forward

        def counting_forward(x):
            w = m.weight
            Cin = w.shape[1]
            Cout = w.shape[0]
            k = w.shape[2] * w.shape[3]
            y = orig(x)
            H_out, W_out = y.shape[-2], y.shape[-1]
            self.mul_op += Cin * Cout * k * H_out * W_out
            self.add_op += (Cin * k - 1) * Cout * H_out * W_out
            return y

        m.forward = counting_forward
        self._patches.append((m, orig))

    def restore(self):
        for m, orig in self._patches:
            m.forward = orig
            if hasattr(m, "_ops_wrapped"):
                delattr(m, "_ops_wrapped")
        self._patches = []

    def __enter__(self):
        self.add_op = 0
        self.mul_op = 0
        self.wrap()
        return self

    def __exit__(self, *args):
        self.restore()


def spike_op_per_step(slot_count, fc_dim, kernel_dims, win, conv_channels,
                      is_conv, n_lif_layers):
    """Rough per-step neuron-internal ops (LIF mem/spike update).

    fc: 3 ops per element per slot per step (mem*decay, (1-spike), +x)
    conv: 3 ops per element per slot per step, summed over conv layers.
    """
    if not is_conv:
        # fc case: 3 * fc_dim per slot per step
        return 3 * slot_count * fc_dim * win
    # conv case: per layer l, kernel_dims[l]^2 * conv_channels[l]
    total = 0
    for k_dim, c_out in zip(kernel_dims, conv_channels):
        total += 3 * slot_count * (k_dim * k_dim) * c_out * win
    return total


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"D:\MyProjects\neuron\complexity\calculate_flops")
    from models.models_4LIF_fc.LIF_1_3_fc import SNN_Model_LIF_1_3

    m = SNN_Model_LIF_1_3(2)
    x = torch.randn(1, 1, 36, 36)
    with OpsTracker(m) as t:
        y = m(x)
    print(f"FC LIF_1_3: add={t.add_op/1e6:.3f}M, mul={t.mul_op/1e6:.3f}M")

    # Cross-check with existing fourLIF_conv
    from models.fourLIF_conv import SCNN_Model_4LIF
    m2 = SCNN_Model_4LIF(2, win=15)
    with OpsTracker(m2) as t2:
        y2 = m2(x)
    print(f"Conv 4LIF: add={t2.add_op/1e6:.3f}M, mul={t2.mul_op/1e6:.3f}M")
