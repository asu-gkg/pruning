import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.nn.utils import prune
import matplotlib.pyplot as plt
from torchvision.models import resnet18
import argparse


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
        sparse = (torch.sum(tensor == 0).item() / tensor.numel()) * 100
        # print(sparse, tensor.shape)
        return tensor

    def decompress(self, compressed_tensor):
        return compressed_tensor


class MySGD(optim.SGD):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(MySGD, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                                    nesterov=nesterov)
        self.compressor = Compressor()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            dampening = group['dampening']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf
                    d_p.add_(weight_decay, p.data)
                else:
                    d_p = d_p.add(p.grad)

                compressed_d_p = self.compressor.compress(d_p)
                p.data.add_(-lr, compressed_d_p)

        return loss


def train_model(epoch, net, trainloader, optimizer, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f'Epoch {epoch}, Loss: {train_loss / len(trainloader)}, Accuracy: {accuracy}')
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--compress', default=False, type=bool, help='compress')
    args = parser.parse_args()

    trainloader, testloader = load_data()
    net = load_resnet18()
    print("Successfully load resnet18 model")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    if args.compress:
        optimizer = MySGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    for epoch in range(5):
        train_acc = train_model(epoch, net, trainloader, optimizer, criterion)
        scheduler.step()
