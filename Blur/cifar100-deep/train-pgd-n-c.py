"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from torchvision import transforms, datasets, models
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torchvision
from torch.utils.data import DataLoader

from pgd.cifar100.model_c import Model
from pgd.cifar100.pgd_attack import LinfPGDAttack
from PIL import Image
#Model4 - n
from tensorflow.python.client import device_lib

with tf.device('/GPU:0'):
    with open('pgd/cifar100/config_cifar100.json') as config_file:
        config = json.load(config_file)
    
    # Setting up training parameters
    tf.set_random_seed(config['random_seed'])
    
    max_num_training_steps = config['max_num_training_steps']
    num_output_steps = config['num_output_steps']
    num_summary_steps = config['num_summary_steps']
    num_checkpoint_steps = config['num_checkpoint_steps']
    eval_on_cpu = config['eval_on_cpu']
    batch_size = config['training_batch_size']
    from_model=0
    # Setting up the data and the model
    img_size = [32,32,3]
    data_root='sar_data/cifar100'
    # Setting up the Tensorboard and checkpoint outputs
    model_name='/Cifar100_Model_c.h5'
    model_dir = "pgd/models/cifar100-n"#config['model_dir-n2']
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
    def get_data(train_batch_size):
        
        data_transform = Compose([Resize((img_size[0], img_size[1])),ToTensor()])
        
        train_loader = DataLoader(datasets.CIFAR100(root=data_root, train=True, transform=data_transform, target_transform=None, download=True),
                                  batch_size=train_batch_size, shuffle=True)
        
        
        return train_loader
    model = Model(True)
    model.model.compile(optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.9, name='SGD'),
        loss='mean_squared_error',metrics=['accuracy'])
    if(from_model==1):
        model.model.load_weights(model_dir+ model_name)
    # Setting up the optimizer
    
    # Set up adversary
    '''if eval_on_cpu:
      with tf.device("/cpu:0"):
          model = Model()
          attack = LinfPGDAttack(model, 
                               .04,#config['epsilon'],
                               config['k'],
                               config['a'],
                               config['random_start'],
                               config['loss_func'])
    else:
        with tf.device("/GPU:0"):
            model = Model()
            attack = LinfPGDAttack(model, 
                                 config['epsilon'],
                                 config['k'],
                                 config['a'],
                                 config['random_start'],
                                 config['loss_func'])'''
    
    # We add accuracy and xent twice so we can easily make three types of
    # comparisons in Tensorboard:
    # - train vs eval (for a single run)
    # - train of different runs
    # - eval of different runs
    
    
    shutil.copy('pgd/cifar100/config_cifar100.json', model_dir)
    
    
    # Initialize the summary writer, global variables, and our time counter.
    #saver.restore(sess,tf.train.latest_checkpoint(model_dir))
    training_time = 0.0
    train_loader= get_data(batch_size)
    trainiter = iter(cycle(train_loader))
    # Main training loop
    for ii in range(max_num_training_steps):
      start = timer()
      x_batch, y_batch = next(trainiter)
      y_batch=np.array(y_batch,dtype='uint8')
      x_batch = np.array(x_batch.data.numpy().transpose(0,2,3,1))
      y_batch_act=np.zeros([len(x_batch),100])
      for i in range(len(x_batch)):
          y_batch_act[i,y_batch[i]]=1
      x_batch_adv = x_batch#attack.perturb(x_batch, y_batch, sess)
      #x_batch_adv_act = attack.perturb(x_batch, y_batch, sess
      model.model.fit(x=x_batch,y=y_batch_act,batch_size=len(x_batch),verbose=0);
      
      # Output to stdout
      if ii % num_output_steps == 0:
        nat_output = model.model.predict(x=x_batch,batch_size=len(x_batch),verbose=1);
        total_correct=0
        for ij in range(len(x_batch)):
            for ik in range (100):
                maxval=0
                maxindex=0
                if(nat_output[ij,ik]>maxval or ik==0):
                    maxval=nat_output[ij,ik]
                    maxindex=ik
            if(maxindex==y_batch[ij]):
                total_correct+=1
        accuracy=total_correct/(len(x_batch))
        print(accuracy)
    
    
      # Actual training step
      model.model.save(model_dir+ model_name)#+"/cifar100-n-2.h5")
      if(ii==38000):
          model = Model(True)
          model.model.compile(optimizer=tf.keras.optimizers.SGD(
              learning_rate=0.001, momentum=0.9, name='SGD'),
              loss='mean_squared_error',metrics=['accuracy'])
          model.model.load_weights(model_dir+ model_name)
      if(ii==48000):
          model = Model(True)
          model.model.compile(optimizer=tf.keras.optimizers.SGD(
              learning_rate=0.0001, momentum=0.9, name='SGD'),
              loss='mean_squared_error',metrics=['accuracy'])
          model.model.load_weights(model_dir+ model_name)
      end = timer()
      training_time += end - start
