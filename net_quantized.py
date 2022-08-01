import torch
import torch.nn as nn
import torch.optim as optim

from archs import MNISTQuantizedNet
from net import MNISTClassifier

random_seed = 1
torch.manual_seed(random_seed)
torch.backends.cudnn.enabled = True
torch.set_printoptions(sci_mode=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device found: {device}')

class MNISTQuantizedClassifier(MNISTClassifier):
    def __init__(self):
        self.net = MNISTQuantizedNet.to(device)
        