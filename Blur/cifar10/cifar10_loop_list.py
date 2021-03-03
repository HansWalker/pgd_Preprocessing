"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import tensorflow as tf

from pgd.cifar10at32.model import Model
from pgd.cifar10at32.pgd_attack import LinfPGDAttack
from torchvision import datasets
from torch.utils.data import DataLoader

from sargan_dep.sargan_models import SARGAN
from sargan_dep.sar_utilities import add_gaussian_noise
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from torchvision.transforms import Compose, ToTensor,  Resize
from PIL import Image
data_root='sar_data/cifar10'


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
def get_data(train_batch_size):
    
    data_transform = Compose([ToTensor()])
    
    train_loader = DataLoader(datasets.CIFAR10(root=data_root, train=True, transform=data_transform, target_transform=None, download=True),
                              batch_size=train_batch_size, shuffle=True)
    
    
    return train_loader
global_step = tf.contrib.framework
# Global constants
with open('pgd/cifar10at32/config_cifar10.json') as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']

model_dir = config['model_dir-n']
# Set upd the data, hyperparameters, and the model


img_size = [32,32,1]
trained_model_path = 'trained_models/sargan_cifar10-x1-4'
trained_model_path2 = 'trained_models/sargan_cifar10-x2-4'
trained_model_path3 = 'trained_models/sargan_cifar10-x3-4'
trained_model_path4 = 'trained_models/sargan_cifar10-x4-4'
BATCH_SIZE = 64
NOISE_STD_RANGE = [0.1, 0.02]

starting_pert=0
ending_pert=.12
current_pert=0
number_of_runs=int(math.ceil(num_eval_examples / eval_batch_size))
change=(ending_pert-starting_pert)/(number_of_runs-1)
attacks=[]
if eval_on_cpu:
  with tf.device("/cpu:0"):
      model = Model()
      for i in range(number_of_runs):
          attacks.append(LinfPGDAttack(model, 
                               current_pert,#config['epsilon'],
                               config['k'],
                               config['a'],
                               config['random_start'],
                               config['loss_func']))
          current_pert+=change
else:
    model = Model()
    for i in range(number_of_runs):
        attacks.append(LinfPGDAttack(model, 
                               current_pert,#config['epsilon'],
                               config['k'],
                               config['a'],
                               config['random_start'],
                               config['loss_func']))
        current_pert+=change

global_step = tf.contrib.framework.get_or_create_global_step()
# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
  os.makedirs(eval_dir)

last_checkpoint_filename = ''
already_seen_state = False
saver = tf.train.Saver()
#summary_writer = tf.summary.FileWriter(eval_dir)
# A function for evaluating a single checkpoint
# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename):
    #sys.stdout = open(os.devnull, 'w')
    g2 =tf.Graph()
    gx2 =tf.Graph()
    gx3 =tf.Graph()
    gx4 =tf.Graph()
    g3=tf.Graph()
    loop_list_adv=np.zeros([2,number_of_runs])
    loop_list_auto=np.zeros([2,number_of_runs])
    epsilonc=starting_pert
    with tf.Session() as sess:
    # Restore the checkpoint
        saver.restore(sess, tf.train.latest_checkpoint(model_dir));
        
        # Iterate over the samples batch-by-batch
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
        x_corr_list=[]
        x_adv_list=[]
        y_batch_list=[]
        
        train_loader= get_data(BATCH_SIZE)
        trainiter = iter(cycle(train_loader))
        for ibatch in range(num_batches):
                    
            x_batch2, y_batch = next(trainiter)
            y_batch_list.append(y_batch)
            x_batch2 = np.array(x_batch2.data.numpy().transpose(0,2,3,1))*255
            x_batch=np.zeros([len(x_batch2),img_size[0]*img_size[1]])
            for i in range(len(x_batch2)):
                nextimage=Image.fromarray((x_batch2[i]).astype(np.uint8))
                nextimage=nextimage.convert('L')
                x_batch[i]=np.array(nextimage,dtype='float32').reshape([img_size[0]*img_size[1]])/255
            x_batch_adv = attacks[ibatch].perturb(x_batch, y_batch, sess)
            x_batch_adv2=np.zeros([len(x_batch),img_size[0],img_size[1],img_size[2]])
            for k in range(len(x_batch)):
                x_batch_adv2[k]=add_gaussian_noise(x_batch_adv[k].reshape([img_size[0],img_size[1],img_size[2]]), sd=np.random.uniform(NOISE_STD_RANGE[1], NOISE_STD_RANGE[1]))
            x_corr_list.append(x_batch_adv)
            x_adv_list.append(x_batch_adv2)
            
      
    with g2.as_default():
        with tf.Session() as sess2:
            sargan_model=SARGAN(img_size, BATCH_SIZE, img_channel=1)
            sargan_saver= tf.train.Saver()    
            sargan_saver = tf.train.import_meta_graph(trained_model_path+'/sargan_mnist.meta');
            sargan_saver.restore(sess2,tf.train.latest_checkpoint(trained_model_path));
            for ibatch in range(num_batches):
                processed_batch=sess2.run(sargan_model.gen_img,feed_dict={sargan_model.image: x_adv_list[ibatch], sargan_model.cond: x_adv_list[ibatch]})
                x_adv_list[ibatch]=processed_batch
    with gx2.as_default():
        with tf.Session() as sessx2:
            sargan_model2=SARGAN(img_size, BATCH_SIZE, img_channel=1)
            sargan_saver2= tf.train.Saver()    
            sargan_saver2 = tf.train.import_meta_graph(trained_model_path2+'/sargan_mnist.meta');
            sargan_saver2.restore(sessx2,tf.train.latest_checkpoint(trained_model_path2));
            for ibatch in range(num_batches):
                processed_batch=sessx2.run(sargan_model2.gen_img,feed_dict={sargan_model2.image: x_adv_list[ibatch], sargan_model2.cond: x_adv_list[ibatch]})
                x_adv_list[ibatch]=(processed_batch)
    with gx3.as_default():
        with tf.Session() as sessx3:
            sargan_model3=SARGAN(img_size, BATCH_SIZE, img_channel=1)
            sargan_saver3= tf.train.Saver()    
            sargan_saver3 = tf.train.import_meta_graph(trained_model_path3+'/sargan_mnist.meta');
            sargan_saver3.restore(sessx3,tf.train.latest_checkpoint(trained_model_path3));
            for ibatch in range(num_batches):
                processed_batch=sessx3.run(sargan_model3.gen_img,feed_dict={sargan_model3.image: x_adv_list[ibatch], sargan_model3.cond: x_adv_list[ibatch]})
                x_adv_list[ibatch]=processed_batch
    with gx4.as_default():
        with tf.Session() as sessx4:
            sargan_model4=SARGAN(img_size, BATCH_SIZE, img_channel=1)
            sargan_saver4= tf.train.Saver()    
            sargan_saver4 = tf.train.import_meta_graph(trained_model_path4+'/sargan_mnist.meta');
            sargan_saver4.restore(sessx4,tf.train.latest_checkpoint(trained_model_path4));
            for ibatch in range(num_batches):
                processed_batch=sessx4.run(sargan_model4.gen_img,feed_dict={sargan_model4.image: x_adv_list[ibatch], sargan_model4.cond: x_adv_list[ibatch]})
                x_adv_list[ibatch]=processed_batch.reshape([len(x_batch),img_size[0]*img_size[1]])           
    with g3.as_default():
        model3 = Model()
        saver2 = tf.train.Saver()
        with tf.Session() as sess3:
            saver2.restore(sess3, filename);
            for ibatch in range(num_batches):
                
                dict_corr = {model3.x_input: x_corr_list[ibatch],
                        model3.y_input: y_batch_list[ibatch]}
                
                dict_adv = {model3.x_input: x_adv_list[ibatch],
                          model3.y_input: y_batch_list[ibatch]}
                
                
        
                
                cur_corr_corr, cur_xent_corr = sess3.run(
                                              [model3.num_correct,model3.xent],
                                              feed_dict = dict_corr)
                
                cur_corr_adv, cur_xent_adv = sess3.run(
                                              [model3.num_correct,model3.xent],
                                              feed_dict = dict_adv)
        
                loop_list_adv[0,ibatch]=epsilonc
                loop_list_adv[1,ibatch]=cur_corr_adv/eval_batch_size
                loop_list_auto[0,ibatch]=epsilonc
                loop_list_auto[1,ibatch]=cur_corr_corr/eval_batch_size
                epsilonc+=change
            '''summary = tf.Summary(value=[
              tf.Summary.Value(tag='xent adv eval', simple_value= avg_xent_adv),
              tf.Summary.Value(tag='xent adv', simple_value= avg_xent_adv),
              tf.Summary.Value(tag='xent nat', simple_value= avg_xent_nat),
              tf.Summary.Value(tag='accuracy adv eval', simple_value= acc_adv),
              tf.Summary.Value(tag='accuracy adv', simple_value= acc_adv),
              tf.Summary.Value(tag='accuracy nat', simple_value= acc_nat)])
            summary_writer.add_summary(summary, global_step.eval(sess3))'''
    #sys.stdout = sys.__stdout__
    return loop_list_adv, loop_list_auto

# Infinite eval loop
while True:
  cur_checkpoint = tf.train.latest_checkpoint(model_dir)

  # Case 1: No checkpoint yet
  if cur_checkpoint is None:
    if not already_seen_state:
      #print('No checkpoint yet, waiting ...', end='')
      already_seen_state = True
    else:
        a=1
      #print('.', end='')
    sys.stdout.flush()
    time.sleep(10)
  # Case 2: Previously unseen checkpoint
  elif cur_checkpoint != last_checkpoint_filename:
    #print('\nCheckpoint {}, evaluating ...   ({})'.format(cur_checkpoint,
    #                                                      datetime.now()))
    sys.stdout.flush()
    last_checkpoint_filename = cur_checkpoint
    already_seen_state = False
    for i in range(75):
        loop_list_adv, loop_list_auto = evaluate_checkpoint(cur_checkpoint)
        np.save('loop_data/cifar10_loop_list'+str(i)+".npy",loop_list_adv)
        np.save('loop_data/cifar10_loop_list_corr'+str(i)+".npy",loop_list_auto)
  # Case 3: Previously evaluated checkpoint
  else:
    if not already_seen_state:
      #print('Waiting for the next checkpoint ...   ({})   '.format(
       #     datetime.now()),
        #    end='')
      already_seen_state = True
    else:
      #print('.', end='')
      a=1
    sys.stdout.flush()
    time.sleep(10)

