"""run.py:"""
#!/usr/bin/env python
from __future__ import print_function
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torchvision.datasets as datasets
from torchvision import transforms, utils
from random import Random
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import time
import math

# # Hyper-parameters
learning_rate = 0.01
nbatch = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    
    # total number of processes to run
    args.world_size = args.gpus * args.nodes    
    
    #IP address for process 0 so that all proc can sync up at first
    os.environ['MASTER_ADDR'] = '10.57.23.164'              
    os.environ['MASTER_PORT'] = '8888'      
    # each process run train(i, args)                
    mp.spawn(train, nprocs=args.gpus, args=(args,))       
        


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x




""" Distributed Synchronous SGD Example """
def train(gpu, args):
    
    #  global rank of the process (one proc per gpu)
    rank = args.nr * args.gpus + gpu
    
    
    # used nccl backend (fastest)             
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=args.world_size,                              
        rank=rank                                               
    ) 
    
    torch.manual_seed(0)    
    model = MLP()
    
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    
    
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    
    # Create a SGD optimizer for gradient descent
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    
    # wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    
    

    train_dataset  = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    
    
    # makes sure that each process gets a different slice of the training data.
    # Use the nn.utils.data.DistributedSampler instead of shuffling the usual way.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    
    start = datetime.now()
    total_step = len(train_loader)  
         
    for epoch in range(args.epochs):
        
        for i, (images, labels) in enumerate(train_loader):
            
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = model(images) # forward pass input through the network
            loss = criterion(outputs, labels)
            
            
            # Backward and optimize
            optimizer.zero_grad() # zeroes the gradient buffers to reset the gradient computed by last images batch
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    loss.item())
                   )
            if idx > 5:
                print("NORMAL END")
                break
                
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == "__main__":
    main()
        
    