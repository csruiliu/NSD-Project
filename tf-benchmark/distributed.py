
import os
import time
import argparse
import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds

from tensorflow.python.keras.layers import *                                                    
from tensorflow.python.keras.models import Model
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline
#from tensorflow.keras.backend import set_session

# Hyper-parameters
learning_rate = 0.001
iteration = 10
#iteration = 100
#nbatch = args.batch_size

# image shapes
#height = args.height
#width  = args.width
#height = 32
#width  = 32
#height = 224
#width  = 224
channel=1

# image label class
num_classes = 10

def main(args):

    # args 
    nbatch = args.batch_size
    height = args.height
    width  = args.width

    # Select model
    #   mlp or mobilenet or resnet
    if args.mtype == 'mlp': 
      from mlp import model_fn
    elif args.mtype == 'mobilenet': 
      from mobilenet import model_fn
    elif args.mtype == 'resnet': 
      from resnet import model_fn

    # Attempting to connect all nodes in `tf.train.ClusterSpec`.
    # Specify your machine's IP and Port
    #cluster_spec = tf.train.ClusterSpec({
    #    'worker': [
    #        '10.143.3.3:2222'
    #    ],
    #    'ps': ['128.135.164.173:55555'],
    #})
    cluster_spec = tf.train.ClusterSpec({
        'worker': args.worker_ip_port,
        'ps': args.ps_ip_port,
    })

    # Specify GPU config here otherwise distributed training won't reflect the config
    config = tf.ConfigProto(
              gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction = 0.6),
              allow_soft_placement=True,
    )

    # Add config option specified above. O.w. PS & Worker server may consume all GPU memory
    server = tf.train.Server(cluster_spec,
                             job_name=args.job_name,
                             task_index=args.task_index,
                             config=config)

    if args.job_name == "ps":
        # `server.join()` means it's NEVER killed
        server.join()
    else:
        # Store the variables into `ps`.
        print("\n   Worker Wake up!   \n", flush=True)
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % args.task_index,
                cluster=cluster_spec)):

            # get data
            if args.dataname == 'mnist':
              ### MNIST
              mnist = input_data.read_data_sets(os.path.abspath(args.datadir), one_hot=True)
              train_images =  mnist.test.images
              train_labels =  mnist.test.labels
              dataset = input_fn(train_images, train_labels,height, width, nbatch, args.mtype)
            else:
              ### Imagenet2012
              datasets = tfds.load('imagenet2012', data_dir=args.datadir)
              dataset  = input_imgnet_fn(datasets, args.nbatch) 

            train_iterator = dataset.make_initializable_iterator()
            imgs, labels = train_iterator.get_next()

            # global step config
            global_step = tf.train.create_global_step()
            get_global_step = tf.train.get_global_step()

            # Model functional
            if args.mtype == 'mlp':
              model = model_fn(input_shape=(height*width,), num_classes=num_classes)
            else :
              model = model_fn(input_shape=(height, width, channel), num_classes=num_classes)

            # prediction
            preds = model(imgs)

            # training options
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=preds))
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=global_step)


            # Initialize the variables, if `is_chief`.
            is_chief = (args.task_index == 0)

            # Terminate hook
            hooks = [tf.train.StopAtStepHook(last_step=iteration),
                     tf.train.CheckpointSaverHook('./example-save',
                                                  save_steps=iteration,
                                                  saver=tf.train.Saver(max_to_keep=1))]
            
            # Training
            with tf.train.MonitoredTrainingSession(
                    is_chief=is_chief,
                    config=config,
                    master=server.target,
                    hooks=hooks) as sess:

                sess.run(train_iterator.initializer)
                start_time = time.perf_counter()
                while not sess.should_stop():
                    step, _, train_loss = sess.run([get_global_step, train_op, loss])
                    print('In {step} step: loss = {loss}'
                          .format(step=step, loss=train_loss))
                stop_time = time.perf_counter()
            stop_time = time.perf_counter()
        
            # logout
            print('  duration (via time.perf_counter()): %f (%f - %f)' % (stop_time - start_time, stop_time, start_time))
            print('  Throughput [image/sec] : %f  ' % (nbatch*iteration/(stop_time - start_time) ) )
            
            # save log
            os.makedirs("./log/"+args.dataname, exist_ok=True)
            with open("./log/"+args.dataname+"/"+args.expname+'.txt', "w") as f:
              f.write(str(stop_time - start_time)+','+str(nbatch*iteration/(stop_time - start_time)))

def input_imgnet_fn(dataset, nbatch=32):
    dataset = dataset.batch(nbatch)
    return dataset


def input_fn(images, labels, height=32, width=32, nbatch=32, model_type='mlp',prefetch=1):
    """ 
    INPUT:  imagess : numpy image [batch, height, width, channel]. np.ndarray
            labels  : numpy image labels.  This should be one-hot 
            height/width : resized height, width
            model_type   : model type
            nbatch : batch size

    OUTPUT: tf.data object
    """
    # reshape other than mlp model
    ndim = len(images.shape)
    print(ndim, flush=True)
    if ndim <= 3:
      # case MNIST, images = (batch, h x w, c)
      if ndim == 2:
        _, hw = np.shape(images)
        c = 1
      elif ndim == 3:
        _, hw, c = np.shape(images)
      h = w = int(np.sqrt(hw))
      images = images.reshape(-1,h,w,c)

    images = tf.image.resize(images, (height, width))
    if model_type == 'mlp':
      images = tf.reshape(images, [-1,height*width])
  
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(nbatch)
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-name', dest='job_name', type=str,
                        choices=['ps', 'worker'])
    parser.add_argument('--ps-ip-port', dest='ps_ip_port', nargs='+')
    parser.add_argument('--worker-ip-port', dest='worker_ip_port', nargs='+')
    parser.add_argument('--task-index', dest='task_index', type=int,
                        default=0)
    parser.add_argument('--mtype', dest='mtype', type=str,
                        default='mlp')
    parser.add_argument('--dataname', dest='dataname', type=str,
                        default='mnist')
    parser.add_argument('--datadir', dest='datadir', type=str,
                        default='./MNIST')
    parser.add_argument('--expname', dest='expname', type=str,
                        default='default')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=32)
    parser.add_argument('--height', dest='height', type=int,
                        default=28)
    parser.add_argument('--width', dest='width', type=int,
                        default=28)
    main(parser.parse_args())
