#!/usr/bin/env python
"""
Reviser: EJYang
    - this code is written upon the given example in the pyTorch tutorial
    - this code runs on multi-gpu multi-node environment
    - this code spwans generate several processes : # of GPUs for the node
    - no error handling was written (GPU check etc.)
"""

import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description="Multinode Test")
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--dist-url', default='127.0.0.1:8885', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--rank', default=0, type=int,
                    help='rank of the process')
parser.add_argument('--GPU-list', default=0, type=str,
                    help='list of node in terms of distributed training ex: --GPU-list=4,4')
parser.add_argument('--node-index', default=0, type=int,
                    help='index of worker node')

class Partition(object):
    """ Dataset-like object, but only access a sbset of it."""

    def __init__(self, data, index):
        self.data = data
        self.index= index

    def __len__(self):
        return len(self.index)


    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chunks """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

class Net(nn.Module):
    """ Network architecture """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def partition_dataset():
    """ Partitioning MNIST """
    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    size = dist.get_world_size()
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size = int(bsz), shuffle=True)

    return train_set, bsz

def average_gradients(model):
    """
    Gradient averaging
    :param model:
    :return:
    """

    size = float(dist.get_world_size())
    for param in model.parameters():
        # using default group in all_reduce!!!
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def run(rank, size):
    """
    Distributed Synchronous SGD Example
    :param rank:
    :param size:
    :return:
    """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    device = torch.device("cuda:%d"%rank)
    model = Net()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum= 0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = Variable(data).to(device), Variable(target).to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.data
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ' , epoch ', epoch, ' : ', epoch_loss / num_batches)

def init_processes(global_rank, local_rank, size, world_size, fn, backend='gloo'):
    """ Initialize the distributed environment """
    dist.init_process_group(backend=args.backend, init_method=args.dist_url,rank=global_rank, world_size=world_size)
    fn(local_rank, size)

if __name__ == "__main__":

    global args
    args = parser.parse_args()

    processes = []
    node_index = args.node_index
    gpu_list = [int(item) for item in args.GPU_list.split(',')]
    local_size = gpu_list[node_index]
    world_size = sum(gpu_list)
    offset = sum(gpu_list[:node_index])

    for rank in range(local_size):
        # global rank is offset+rank / local rank is rank
        p = Process(target=init_processes, args=(offset+rank, rank, local_size, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
