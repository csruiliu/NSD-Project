#!/bin/bash

# Basic info 
# + Data = {MNIST, ImageNet}
#
# + Iteration = 100 
#

## Config
job_name='worker'
task_index=0
ps_ip_port=("128.135.164.173:22222")
worker_ip_port=("10.143.3.3:22222")
#worker_ip_port=("10.143.3.3:22222" "10.150.49.25:22222")
network_name="enp4s0" # your network name

## python path
# * MUST specify since we run by root
#python_path="python"
python_path="/home/tkurihana/.conda/envs/tf-gpu/bin/python"
pytorch_path="/home/tkurihana/.conda/envs/pytorch/bin/python"

## Dataset
#dataname='mnist' # or imagenet
#datadir='./MNIST' 
dataname='imagenet' # or imagenet
datadir='./Imagenet/imagenet1k' 

## Height/Width
height=224
width=224
# Mobilenet will accept 32 or larger Height/Widht
#height=32
#width=32

## Lists
#models=("mlp" "mobilenet" "resnet")
models=("mlp")
batch_sizes=(32)
#batch_sizes=(32 64 128)

## EXP config list
delays=(0.1)  #ms
bandwidths=(1000)  #Mbits
packets=(0) # percent loss
#delays=(1 10 100)  #ms
#bandwidths=(1000 500 100)  #Mbits(FUll ; 1/2 ; 1/10)
#bandwidths=(500)  #Mbits(FUll ; 1/2 ; 1/10)
#packets=(0 1 10) # percent loss

echo 'START PERFORMANCE EXP'

if  [ ${job_name} = "ps" ]; then
  echo "PS"
  $python_path distributed.py \
        --job-name ${job_name} --task-index ${task_index} --mtype "mlp" \
        --ps-ip-port ${ps_ip_port} \
        --worker-ip-port ${worker_ip_port} \
        --dataname ${dataname} \
        --datadir ${datadir} \
        --expname "param-server"  --batch_size 1 --height 1 --width 1
fi
if  [ ${job_name} = "worker" ]; then
  echo "WORKER"
  # initial settings
  echo `sudo tc qdisc add dev ${network_name} root netem delay 1ms`
  sleep 10

  for model in "${models[@]}" ; do
    for delay in "${delays[@]}" ; do
      for bandwidth in "${bandwidths[@]}" ; do
        for packet in "${packets[@]}" ; do
          for batch_size in "${batch_sizes[@]}" ; do
            # START
            cexpname="${model}_${batch_size}-${packet}ps-${bandwidth}mbit-${delay}ms"

            echo `sudo tc qdisc change dev ${network_name} root netem delay ${delay} loss ${packet}% rate ${bandwidth}mbit `


            echo "CLEAN CUDA CACHE"
            $pytorch_path  clean.py
            echo rm -fr __pycache__

            $python_path distributed.py \
                  --job-name ${job_name} --task-index ${task_index} --mtype ${model} \
                  --ps-ip-port ${ps_ip_port} \
                  --worker-ip-port ${worker_ip_port} \
                  --dataname ${dataname} \
                  --datadir ${datadir} \
                  --expname ${cexpname}  --batch_size ${batch_size} --height ${height} --width ${width} 
            # END
            sleep 5
          done
        done
      done
    done
  done
  echo `sudo tc qdisc del dev ${network_name} root`
fi

echo 'NORMAL END'

