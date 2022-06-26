import torch
import torch.nn as nn

class Conv2DBlock(nn.Module):
    def __init__(self, in_chs, out_chs, ksize, stride, padding):
        super(Conv2DBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, ksize, stride, padding, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.block(x)

# Straight-Through Estimator (STE)
class DiscreteFunction(torch.autograd.Function):
    def __init__(self, func):
        super(DiscreteFunction, self).__init__()
        self.func = func

    def forward(self, ctx, input):
        return self.func(input)

    def backward(self, ctx, output_grad):
        # clip in [-1, 1]
        return torch.nn.HardTanh(output_grad)

class Quantizer(nn.Module):
    def __init__(self, min_val, max_val, is_signed=False):
        super(Quantizer, self).__init__()

        self.is_signed = is_signed

        self.scale = torch.nn.Parameter(max_val - min_val, dtype=torch.int8)
        self.zero_pnt = torch.nn.Paramter(0, dtype=torch.int8)
        self.bit_depth = torch.nn.Paramter(8, dtype=torch.int8)

    def forward(self, x):
        quantize = lambda x: torch.clamp()
        ste = DiscreteFunction(quantize)
        val = 2 << self.bit_depth
        half_val >>= 2
        return ste.forward(quantize, -half_val - 1, half_val - 1) if self.is_signed \
            else ste.forward(quantize, 0, val - 1)
