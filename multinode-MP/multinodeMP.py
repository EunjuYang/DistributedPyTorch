#!/usr/bin/env python
"""
Writer: EJYang
    - Model parallelism test in the one node with 2 GPUs
"""

from __future__ import print_function
from random import Random
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.multiprocessing import Process
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

parser = argparse.ArgumentParser(description="Multinode Test")
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
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
parser.add_argument('--batch-size', default=1024, type=int,
                    help='batch size')
parser.add_argument('--IP', default='127.0.0.1', type=str,
                    help='IP address for master')
parser.add_argument('--portNum', default='39990', type=str,
                    help='port Num for master')
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

class Net1(nn.Module):
    """ Network architecture """

    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        return x

class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def run(global_rank, local_rank, world_size):

    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])
    )
    batch_size = args.batch_size
    train_set = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    if global_rank == 0:
        net1  = Net1().cuda(local_rank)
        optimizer1 = optim.SGD(net1.parameters(), lr=0.01)

    elif global_rank == 1:
        net2  = Net2().cuda(local_rank)
        optimizer2 = optim.SGD(net2.parameters(), lr=0.01)

    for epoch in range(100) :

        for batch_idx, (data, target) in enumerate(train_set):

            if global_rank == 0:

                # for GPU 0
                net1.zero_grad()
                data = Variable(data).cuda(local_rank)

                # forward
                net1_output = net1(data)
                # copy to cpu to send the data
                net1_output_cpu = net1_output.data.cpu()
                dist.send(net1_output_cpu, dst=1)

                #backprop
                from_net2_backprop = torch.FloatTensor(data.size()[0], 320)
                dist.recv(from_net2_backprop, src=1)
                net1_output.backward(from_net2_backprop.cuda(local_rank))
                optimizer1.step()

            if global_rank == 1:

                # for GPU 1
                target = Variable(target).cuda(local_rank)
                from_net1_forward = torch.FloatTensor(target.size()[0], 320)
                dist.recv(from_net1_forward, src=0)
                # copy to GPU 1 from CPU
                net2_input = Variable(from_net1_forward.cuda(local_rank), requires_grad = True).cuda(local_rank)
                #forward for GPU1
                net2_output = net2(net2_input).cuda(local_rank)

                #backprop
                loss = F.nll_loss(net2_output, target).cuda(local_rank)
                net2.zero_grad()
                loss.backward()
                optimizer2.step()

                #backprop
                from_net2_backprop = net2_input.grad.data.cpu()
                dist.send(from_net2_backprop, dst=0)

        if global_rank == 1:
            print('epoch ', epoch, ' : ', loss )

def init_processes(local_rank, global_rank, world_size, fn, backend='gloo'):

    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.IP
    os.environ['MASTER_PORT'] = args.portNum
    dist.init_process_group(backend, init_method="tcp://%s:%s"%(args.IP, args.portNum), rank=global_rank, world_size=world_size)
    fn(global_rank, local_rank, world_size)

if __name__ == "__main__":

    global args
    args = parser.parse_args()

    processes = []
    node_index = args.node_index
    gpu_list = [int(item) for item in args.GPU_list.split(',')]
    local_size = gpu_list[node_index]
    world_size = sum(gpu_list)
    offset = sum(gpu_list[:node_index])

    for local_rank in range(local_size):
        # global rank is offset+rank / local rank is rank
        p = Process(target=init_processes, args=(local_rank, offset+local_rank, world_size, run, args.backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



