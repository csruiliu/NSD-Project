#!/bin/bash

# Basic info 
# + Data = {MNIST, ImageNet}
#
# + Iteration = 100 
#

## Config
job_name="worker"
task_index=0
height=28
width=28
# Mobilenet
#height=32
#width=32

## Lists
#models=("mlp" "mobilenet" "resnet")
models=("mlp")
batch_sizes=(32 128 1024)

## EXP config list
delays=(1 10 100)  #ms
bandwidths=(1000 100 10)  #Mbits
packets=(0 1 10) # percent loss

echo 'START PERFORMANCE EXP'

# initial settings
echo `sudo tc qdisc add dev enp4s0 root netem delay 1ms`
sleep 10

for model in "${models[@]}" ; do
  for delay in "${delays[@]}" ; do
    for bandwidth in "${bandwidths[@]}" ; do
      for packet in "${packets[@]}" ; do
        for batch_size in "${batch_sizes[@]}" ; do
          # START
          cexpname="${model}_${batch_size}-${packet}ps-${bandwidth}mbit-${delay}ms"

          echo `sudo tc qdisc change dev enp4s0 root netem delay ${delay} loss ${packet}% rate ${bandwidth}mbit `

          /home/tkurihana/.conda/envs/tf-gpu/bin/python distributed.py \
                --job-name=${job_name} --task-index=${task_index} --mtype=${model} \
                --expname=${cexpname}  --batch_size=${batch_size} --height=${height} --width=${width} 
          # END
          sleep 5
        done
      done
    done
  done
done


echo `sudo tc qdisc del dev enp4s0 root`
echo 'NORMAL END'

