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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-batch_size', default=32, type=int, metavar='N',
                        help='number of batch size')
    parser.add_argument('-expname', default='default', type=str, metavar='N',
                        help='name of running ')
    args = parser.parse_args()
    
    # total number of processes to run
    args.world_size = args.gpus * args.nodes    
    
    #IP address for process 0 so that all proc can sync up at first
    os.environ['MASTER_ADDR'] = '10.143.3.3'
    os.environ['MASTER_PORT'] = '2223'      
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
    
    # join cpu or cpu 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    #torch.cuda.set_device(gpu)

    if torch.cuda.is_available():
      model.cuda(gpu)
      criterion = nn.CrossEntropyLoss().cuda(gpu)
    else:
      criterion = nn.CrossEntropyLoss()
    
    # Create a Adam optimizer for gradient descent
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    
    # wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[device])
                                                #device_ids=[gpu])
    
    

    # MNIST
    #train_dataset  = datasets.MNIST('./data', train=True, download=True,
    #                         transform=transforms.Compose([
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((0.1307,), (0.3081,))
    #                         ]))

    # Imagnet
    #train_dataset  = datasets.ImageNet('./data', train=True,
    #                         transform=transforms.Compose([
    #                             transforms.Resize(256),
    #                             transforms.RandomCrop(224),
    #                             transforms.RandomHorizontalFlip(),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((0.485, 0.456, 0.406),
    #                                                    (0.229, 0.224, 0.225))
    #                         ]))
    
    
    # makes sure that each process gets a different slice of the training data.
    # Use the nn.utils.data.DistributedSampler instead of shuffling the usual way.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    
    start_time = time.perf_counter()
    total_step = len(train_loader)  
         
    for epoch in range(args.epochs):
        
        for i, (images, labels) in enumerate(train_loader):
            
            if torch.cuda.is_available():
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
            # debug
            #if i > 5:
            #    print("NORMAL END")
            #    break
                
    if gpu == 0:
        stop_time = time.perf_counter()

        # logout
        print('  duration (via time.perf_counter()): %f (%f - %f)' % (stop_time - start_time, stop_time, start_time))
        print('  Throughput [image/sec] : %f  ' % (args.batch_size*total_step*args.epochs/(stop_time - start_time) ) )
            
        # save log
        os.makedirs("./log", exist_ok=True)
        with open("./log/"+args.expname+'.txt', "w") as f:
          f.write(str(stop_time - start_time)+','+str(args.batch_size*total_step*args.epochs/(stop_time - start_time)))


if __name__ == "__main__":
    main()
        
    
