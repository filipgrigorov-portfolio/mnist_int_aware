import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as tds

from archs import MNISTNet
from torch.utils.data import DataLoader, random_split

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(1)
torch.set_printoptions(sci_mode=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device found: {device}')

DEBUG = False
PRINT_EVERY = 500

def timeit(func):
    start_time = time.time()
    def inner(x):
        return func(x)
    return time.time() - start_time
    
class MNISTClassifier:
    def __init__(self):
        self.net = MNISTNet().to(device)

    def load_pretrained(self, path):
        self.net.load_state_dict(torch.load(path))

    def split_data(self):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_data = tds.MNIST(root='data/', train=True, download=True, transform=transform_ops)
        val_data = tds.MNIST(root='data/', train=False, download=True, transform=transform_ops)
        val_data_len = int(0.2 * len(val_data))
        splits = random_split(val_data, [ val_data_len, len(val_data) - val_data_len ])
        val_dat, test_data = splits[0], splits[1]
        
        return train_data, val_data, test_data

    def train(self, nepochs, batch_size, lr, momentum, train_data, val_data):
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, pin_memory=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr, momentum)

        avg_train_loss = 0.0
        avg_train_accuracy = 0.0
        for epoch in range(nepochs):
            for bidx, (x, y_gt) in enumerate(train_loader):
                start_time = time.time()

                x = x.float().to(device)
                y_gt = y_gt.to(device)

                optimizer.zero_grad()

                y_pred = self.net(x)

                loss = criterion(y_pred, y_gt) # avg per batch
                avg_train_loss += loss.item() # detach the value (so so expensive operation)
                train_accuracy = float((torch.argmax(y_pred, dim=1) == y_gt).sum().item()) / batch_size
                avg_train_accuracy += train_accuracy

                if (bidx + 1) % PRINT_EVERY == 0:
                    end_time = time.time()
                    print(f'Training:: [{epoch}][{(bidx + 1) * batch_size} / {len(train_loader) * batch_size}]: '
                          f'loss_i={round(loss.item(), 4)}  -  loss_avg={round(avg_train_loss / (bidx + 1), 4)}  -  '
                          f'acc_i={round(train_accuracy, 4)}  -  '
                          f'acc_avg={round(avg_train_accuracy / (bidx + 1), 4)}  -  '
                            f'time_i={round((end_time - start_time) * 1e3, 4)} ms')
                    
                    self.net.eval()
                    with torch.no_grad():
                        val_loss = 0.0
                        val_accuracy = 0.0
                        for vbidx, (val_x, val_gt_y) in enumerate(val_loader):
                            val_x = val_x.float().to(device)
                            val_gt_y = val_gt_y.to(device)
                            y_pred = self.net(val_x)
                            val_loss += criterion(y_pred, val_gt_y).item()
                            val_accuracy += float((torch.argmax(y_pred, dim=1) == val_gt_y).sum().item()) / batch_size
                        print(f'Validation:: [{(vbidx + 1) * batch_size} / {len(val_loader) * batch_size}]: '
                          f'val_loss_avg={round(val_loss / len(val_loader), 4)}  -  val_acc_avg={round(val_accuracy / len(val_loader), 4)}')
                    self.net.train()

                loss.backward()

                optimizer.step()
                
            print(f'Saving model')
            torch.save(self.net.state_dict(), 'mnist.pth')
            model_scripted = torch.jit.script(self.net)
            model_scripted.save('mnist_scripted.pt')
        
        print(f'\nTraining is complete\n')

    def infer(self, x):
        self.net(x)
        return torch.argmax(self.net(x), dim=1)
