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

from pgd.mnist.model2 import Model
from pgd.mnist.pgd_attack import LinfPGDAttack
from PIL import Image
with open('pgd/mnist/config_mnist.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
eval_on_cpu = config['eval_on_cpu']
batch_size = config['training_batch_size']

# Setting up the data and the model
img_size = [28,28,1]
data_root='sar_data/MNIST'
def cycle(iterable):
    while True:
        for x in iterable:
            yield x
def get_data(train_batch_size):
    
    data_transform = Compose([Resize((img_size[0], img_size[1])),ToTensor()])
    
    train_loader = DataLoader(datasets.MNIST(root=data_root, train=True, transform=data_transform, target_transform=None, download=True),
                              batch_size=train_batch_size, shuffle=True)
    
    
    return train_loader
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(True)
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

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir-n-2']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs


shutil.copy('pgd/mnist/config_mnist.json', model_dir)


# Initialize the summary writer, global variables, and our time counter.
#saver.restore(sess,tf.train.latest_checkpoint(model_dir))
training_time = 0.0
train_loader= get_data(batch_size)
trainiter = iter(cycle(train_loader))
# Main training loop
for ii in range(max_num_training_steps):
  start = timer()
  x_batch2, y_batch = next(trainiter)
  y_batch=np.array(y_batch,dtype='uint8')
  x_batch2 = np.array(x_batch2.data.numpy().transpose(0,2,3,1))
  x_batch=np.zeros([len(x_batch2),img_size[0]*img_size[1]])
  y_batch_act=np.zeros([len(x_batch),10])
  for i in range(len(x_batch2)):
      x_batch[i]=x_batch2[i].reshape([img_size[0]*img_size[1]])
      y_batch_act[i,y_batch[i]]=1
  # Compute Adversarial Perturbations
  x_batch=x_batch.reshape([batch_size,img_size[0],img_size[1],img_size[2]])
  x_batch_adv = x_batch#attack.perturb(x_batch, y_batch, sess)
  #x_batch_adv_act = attack.perturb(x_batch, y_batch, sess
  model.model.fit(x=x_batch,y=y_batch_act,batch_size=batch_size,verbose=0);
  # Output to stdout
  if ii % num_output_steps == 0:
    nat_output = model.model.evaluate(x=x_batch,y=y_batch_act,batch_size=batch_size);
    print('    training nat accuracy ',nat_output * 100)
    #print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
    #if ii != 0:
     # print('    {} examples per second'.format(
      #    num_output_steps * batch_size / training_time))
      #training_time = 0.0
  # Tensorboard summaries
  #if ii % num_summary_steps == 0:
    #summary = sess.run(merged_summaries, feed_dict=adv_dict)
    #summary_writer.add_summary(summary, global_step.eval(sess))


  # Actual training step
  end = timer()
  training_time += end - start
