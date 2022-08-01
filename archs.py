import torch.nn as nn

from layers import Conv2DBlock

class MNISTNet(nn.Module):
    def __init__(self, num_classes = 10, is_strided=False):
        super(MNISTNet, self).__init__()
        
        self.num_classes = num_classes
        
        self.model_head = nn.Sequential(
            # 1x28x28
            Conv2DBlock(1, 6, 5, 1, 0),
            #6x24x24
            Conv2DBlock(6, 12, 5, 1, 0),
            #12x20x20
            Conv2DBlock(12, 24, 5, 1, 0),
            #24x16x16
            Conv2DBlock(24, 32, 3, 1, 0),
            #32x14x14
            nn.MaxPool2d(2, 2),
        )
        
        self.model_tail = nn.Sequential(
            #32x7x7
            Conv2DBlock(32, self.num_classes, 3, 4, 0),
            #nclassesx2x2
            nn.MaxPool2d(2, 2),
            #64x1x1
        ) \
        if is_strided else nn.Sequential(
            Conv2DBlock(32, 64, 3, 1, 1),
            Conv2DBlock(64, 64, 3, 1, 0),
            Conv2DBlock(64, 64, 3, 1, 0),
            Conv2DBlock(64, self.num_classes, 3, 1, 0),
            #nclassesx1x1
        )
        
    def forward(self, x):        
        return self.model_tail(self.model_head(x)).view(-1, self.num_classes)

# Note: QAT network (learns to model and compensate for quantization noise introduced)
class MNISTQuantizedNet(nn.Module):
    def __init__(self):
        pass

# TODO: Add distillation network over the frozen model
