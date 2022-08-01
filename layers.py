from turtle import forward
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
class Quantize(torch.autograd.Function):
    def __init__(self, vmin, vmax, qmin, qmax):
        super(Quantize, self).__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.qmin = qmin
        self.qmax = qmax
        vrange = self.vmax - self.vmin
        qrange = self.qmin - self.qmax
        self.s = vrange / qrange
        self.z = round((vmax * qmin - vmin * qmax) / vrange)

        print('s: ', self.s)
        print('z: ', self.z)

    def forward(self, ctx, input):
        #ctx.save_for_backward(input)
        return torch.clamp(torch.round(self.s * input) - self.z, self.qmin, self.qmax)

    def backward(self, ctx, output_grad):
        #result = ctx.saved_tensors
        #gradient = output_grad * result
        # clip in [-1, 1]
        #return torch.nn.HardTanh(output_grad)
        return 1.0 if output_grad >= -1 and output_grad <= 1 else 0.0

class DeQuantize(torch.autograd.Function):
    def __init__(self, vmin, vmax, qmin, qmax):
        super(DeQuantize, self).__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.qmin = qmin
        self.qmax = qmax
        vrange = self.vmax - self.vmin
        qrange = self.qmin - self.qmax
        self.s = vrange / qrange
        self.z = round((vmax * qmin - vmin * qmax) / vrange)        

    def forward(self, ctx, input):
        return torch.float32((input - self.z) * self.s)

    def backward(self, ctx, output_grad):
        return torch.clip(output_grad, -1, 1)

if __name__ == '__main__':
    bit = 8
    qmin = -(1 << (bit - 1)) + 1
    qmax = (1 << (bit -1)) - 1
    print(f'[{qmin}, {qmax}]')

    a = torch.FloatTensor([3.45, 5.434563, 1.23343])
    b = torch.FloatTensor([2.45, 9.934563, 3.25843])
    c = torch.multiply(a, b)
    print(c)

    cmin = 0; cmax = 100
    amin = 0; amax = 10
    bmin = 0; bmax = 10

    quantize = Quantize(amin, amax, qmin, qmax)
    aq = quantize.forward(ctx=None, input=a)
    dequantize = Quantize(bmin, bmax, qmin, qmax)
    bq = dequantize.forward(ctx=None, input=b)

    print(aq)
    print(bq)
