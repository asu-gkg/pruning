import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.nn.utils import prune
import matplotlib.pyplot as plt
from torchvision.models import resnet18


def load_data():
    # Transform and DataLoader for CIFAR-10
    transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader


def load_resnet18():
    # Load the pretrained ResNet18 model
    net = resnet18(pretrained=True)
    # Replace the final fully connected layer
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 10)
    net = net.to('cuda')
    return net


class Compressor:
    def __init__(self):
        print("Compressor Init")

    def compress(self, tensor):
        print("I'm compressed!")
        return tensor

    def decompress(self, compressed_tensor):
        return compressed_tensor


class MySGD(optim.SGD):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(CompressedSGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.compressor = Compressor()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                compressed_d_p = self.compressor.compress(d_p)
                p.data.add_(-group['lr'], compressed_d_p)

        return loss

def train_model(net, testloader):
    pass

if __name__ == '__main__':
    trainloader, testloader = load_data()
    net = load_resnet18()


    print("Successfully load resnet18 model")
