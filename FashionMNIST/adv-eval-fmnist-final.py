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

from pgd.fmnist.model import Model
from pgd.fmnist.pgd_attack import LinfPGDAttack
from torchvision import datasets
from torch.utils.data import DataLoader

from sargan_dep.sargan_models import SARGAN
from sargan_dep.sar_utilities import add_gaussian_noise
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from torchvision.transforms import Compose, ToTensor,  Resize
from PIL import Image
data_root='sar_data/FMNIST'

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
def get_data(train_batch_size):
    
    data_transform = Compose([ToTensor()])
    
    train_loader = DataLoader(datasets.FashionMNIST(root=data_root, train=False, transform=data_transform, target_transform=None, download=True),
                              batch_size=train_batch_size, shuffle=True)
    
    
    return train_loader
global_step = tf.contrib.framework
# Global constants
with open('pgd/fmnist/config_fmnist.json') as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']

model_dir = config['model_dir']
# Set upd the data, hyperparameters, and the model


img_size = [28,28,1]
trained_model_path = 'trained_models/sargan_fmnist-x1-1'
trained_model_path2 = 'trained_models/sargan_fmnist-x2-1'
trained_model_path3 = 'trained_models/sargan_fmnist-x3-1'
trained_model_path4 = 'trained_models/sargan_fmnist-x4-1'
BATCH_SIZE = 64
NOISE_STD_RANGE = [0.0, 0.2]

if eval_on_cpu:
  with tf.device("/cpu:0"):
      model = Model()
      attack = LinfPGDAttack(model, 
                           .08,#config['epsilon'],
                           config['k'],
                           config['a'],
                           config['random_start'],
                           config['loss_func'])
else:
    model = Model()
    attack = LinfPGDAttack(model, 
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])

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
    #Different graphs for all the models
    gx1 =tf.Graph()
    gx2 =tf.Graph()
    gx3 =tf.Graph()
    gx4 =tf.Graph()
    g3=tf.Graph()
    with tf.Session() as sess:
    # Restore the checkpoint
        saver.restore(sess, filename);
    
        # Iterate over the samples batch-by-batch
        #number of batches
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
        total_xent_nat = 0.
        total_xent_corr = 0.
        total_corr_nat = 0.
        total_corr_corr = 0.
        
        total_corr_adv = np.zeros([4]).astype(dtype='float32')
        total_corr_blur = np.zeros([4]).astype(dtype='float32')
        total_xent_adv = np.zeros([4]).astype(dtype='float32')
        total_xent_blur = np.zeros([4]).astype(dtype='float32')
        
        #storing the various images
        x_batch_list=[]
        x_corr_list=[]
        x_blur_list1=[]
        x_adv_list1=[]
        x_blur_list2=[]
        x_adv_list2=[]
        x_blur_list3=[]
        x_adv_list3=[]
        x_blur_list4=[]
        x_adv_list4=[]
        #Storing y values
        y_batch_list=[]
        
        train_loader= get_data(BATCH_SIZE)
        trainiter = iter(cycle(train_loader))
        for ibatch in range(num_batches):
                
            x_batch2, y_batch = next(trainiter)
            y_batch_list.append(y_batch)
            x_batch2 = np.array(x_batch2.data.numpy().transpose(0,2,3,1))
            x_batch=np.zeros([len(x_batch2),img_size[0]*img_size[1]])
            for i in range(len(x_batch2)):
                x_batch[i]=x_batch2[i].reshape([img_size[0]*img_size[1]])
            x_batch_adv = attack.perturb(x_batch, y_batch, sess)

            x_batch2=np.zeros([len(x_batch),img_size[0],img_size[1],img_size[2]])
            x_batch_adv2=np.zeros([len(x_batch),img_size[0],img_size[1],img_size[2]])
            for k in range(len(x_batch)):
                x_batch2[k]=add_gaussian_noise(x_batch[k].reshape([img_size[0],img_size[1],img_size[2]]), sd=np.random.uniform(NOISE_STD_RANGE[1], NOISE_STD_RANGE[1]))
                x_batch_adv2[k]=add_gaussian_noise(x_batch_adv[k].reshape([img_size[0],img_size[1],img_size[2]]), sd=np.random.uniform(NOISE_STD_RANGE[1], NOISE_STD_RANGE[1]))
            
            x_batch_list.append(x_batch)
            x_corr_list.append(x_batch_adv)
            x_blur_list4.append(x_batch2)
            x_adv_list4.append(x_batch_adv2)
            
    #Running through first autoencoder
    with gx1.as_default():
        with tf.Session() as sess2:
            sargan_model=SARGAN(img_size, BATCH_SIZE, img_channel=1)
            sargan_saver= tf.train.Saver()    
            sargan_saver = tf.train.import_meta_graph(trained_model_path+'/sargan_mnist.meta');
            sargan_saver.restore(sess2,tf.train.latest_checkpoint(trained_model_path));
            for ibatch in range(num_batches):
                processed_batch=sess2.run(sargan_model.gen_img,feed_dict={sargan_model.image: x_adv_list4[ibatch], sargan_model.cond: x_adv_list4[ibatch]})
                x_adv_list4[ibatch]=processed_batch
                
                blurred_batch=sess2.run(sargan_model.gen_img,feed_dict={sargan_model.image: x_blur_list4[ibatch], sargan_model.cond: x_blur_list4[ibatch]})
                x_blur_list4[ibatch]=blurred_batch
                
                #adding images to first autoencoder data set
                x_blur_list1.append(blurred_batch.reshape([len(x_batch),img_size[0]*img_size[1]]))
                x_adv_list1.append(processed_batch.reshape([len(x_batch),img_size[0]*img_size[1]]))
    with gx2.as_default():
        with tf.Session() as sessx2:
            sargan_model2=SARGAN(img_size, BATCH_SIZE, img_channel=1)
            sargan_saver2= tf.train.Saver()    
            sargan_saver2 = tf.train.import_meta_graph(trained_model_path2+'/sargan_mnist.meta');
            sargan_saver2.restore(sessx2,tf.train.latest_checkpoint(trained_model_path2));
            for ibatch in range(num_batches):
                processed_batch=sessx2.run(sargan_model2.gen_img,feed_dict={sargan_model2.image: x_adv_list4[ibatch], sargan_model2.cond: x_adv_list4[ibatch]})
                x_adv_list4[ibatch]=(processed_batch)
                
                blurred_batch=sessx2.run(sargan_model2.gen_img,feed_dict={sargan_model2.image: x_blur_list4[ibatch], sargan_model2.cond: x_blur_list4[ibatch]})
                x_blur_list4[ibatch]=(blurred_batch)
                
                #adding images to second autoencoder data set
                x_blur_list2.append(blurred_batch.reshape([len(x_batch),img_size[0]*img_size[1]]))
                x_adv_list2.append(processed_batch.reshape([len(x_batch),img_size[0]*img_size[1]]))
    with gx3.as_default():
        with tf.Session() as sessx3:
            sargan_model3=SARGAN(img_size, BATCH_SIZE, img_channel=1)
            sargan_saver3= tf.train.Saver()    
            sargan_saver3 = tf.train.import_meta_graph(trained_model_path3+'/sargan_mnist.meta');
            sargan_saver3.restore(sessx3,tf.train.latest_checkpoint(trained_model_path3));
            for ibatch in range(num_batches):
                processed_batch=sessx3.run(sargan_model3.gen_img,feed_dict={sargan_model3.image: x_adv_list4[ibatch], sargan_model3.cond: x_adv_list4[ibatch]})
                x_adv_list4[ibatch]=processed_batch
                
                blurred_batch=sessx3.run(sargan_model3.gen_img,feed_dict={sargan_model3.image: x_blur_list4[ibatch], sargan_model3.cond: x_blur_list4[ibatch]})
                x_blur_list4[ibatch]=blurred_batch
                
                #adding images to third autoencoder data set
                x_blur_list3.append(blurred_batch.reshape([len(x_batch),img_size[0]*img_size[1]]))
                x_adv_list3.append(processed_batch.reshape([len(x_batch),img_size[0]*img_size[1]]))
    #Final autoencoder setup
    with gx4.as_default():
        with tf.Session() as sessx4:
            sargan_model4=SARGAN(img_size, BATCH_SIZE, img_channel=1)
            sargan_saver4= tf.train.Saver()    
            sargan_saver4 = tf.train.import_meta_graph(trained_model_path4+'/sargan_mnist.meta');
            sargan_saver4.restore(sessx4,tf.train.latest_checkpoint(trained_model_path4));
            for ibatch in range(num_batches):
                processed_batch=sessx4.run(sargan_model4.gen_img,feed_dict={sargan_model4.image: x_adv_list4[ibatch], sargan_model4.cond: x_adv_list4[ibatch]})
                x_adv_list4[ibatch]=processed_batch.reshape([len(x_batch),img_size[0]*img_size[1]])
                
                blurred_batch=sessx4.run(sargan_model4.gen_img,feed_dict={sargan_model4.image: x_blur_list4[ibatch], sargan_model4.cond: x_blur_list4[ibatch]})
                x_blur_list4[ibatch]=blurred_batch.reshape([len(x_batch),img_size[0]*img_size[1]])
                
    with g3.as_default():
        model3 = Model()
        saver2 = tf.train.Saver()
        with tf.Session() as sess3:
            saver2.restore(sess3, filename);
            for ibatch in range(num_batches):
                cur_xent_adv = np.zeros([4]).astype(dtype='float32')
                cur_xent_blur = np.zeros([4]).astype(dtype='float32')
                cur_corr_adv = np.zeros([4]).astype(dtype='float32')
                cur_corr_blur = np.zeros([4]).astype(dtype='float32')
                
                dict_nat = {model3.x_input: x_batch_list[ibatch],
                        model3.y_input: y_batch_list[ibatch]}
                
                dict_corr = {model3.x_input: x_corr_list[ibatch],
                        model3.y_input: y_batch_list[ibatch]}
                
                
                #First autoencoder dictionary
                dict_adv1 = {model3.x_input: x_adv_list1[ibatch],
                          model3.y_input: y_batch_list[ibatch]}
                
                dict_blur1 = {model3.x_input: x_blur_list1[ibatch],
                          model3.y_input: y_batch_list[ibatch]}
                
                
                #Second autoencoder dictionary
                dict_adv2 = {model3.x_input: x_adv_list2[ibatch],
                          model3.y_input: y_batch_list[ibatch]}
                
                dict_blur2 = {model3.x_input: x_blur_list2[ibatch],
                          model3.y_input: y_batch_list[ibatch]}
                
                
                #Third autoencoder dictionary
                dict_adv3 = {model3.x_input: x_adv_list3[ibatch],
                          model3.y_input: y_batch_list[ibatch]}
                
                dict_blur3 = {model3.x_input: x_blur_list3[ibatch],
                          model3.y_input: y_batch_list[ibatch]}
                
                
                #Fourth autoencoder dictionary
                dict_adv4 = {model3.x_input: x_adv_list4[ibatch],
                          model3.y_input: y_batch_list[ibatch]}
                
                dict_blur4 = {model3.x_input: x_blur_list4[ibatch],
                          model3.y_input: y_batch_list[ibatch]}
                
                
                #Regular Images
                cur_corr_nat, cur_xent_nat = sess3.run(
                                              [model3.num_correct,model3.xent],
                                              feed_dict = dict_nat)
                
                cur_corr_corr, cur_xent_corr = sess3.run(
                                              [model3.num_correct,model3.xent],
                                              feed_dict = dict_corr)
                
                #First autoencoder dictionary
                cur_corr_blur[0], cur_xent_blur[0] = sess3.run(
                                              [model3.num_correct,model3.xent],
                                              feed_dict = dict_blur1)
                
                cur_corr_adv[0], cur_xent_adv[0] = sess3.run(
                                              [model3.num_correct,model3.xent],
                                              feed_dict = dict_adv1)
                
                #Second autoencoder dictionary
                cur_corr_blur[1], cur_xent_blur[1] = sess3.run(
                                              [model3.num_correct,model3.xent],
                                              feed_dict = dict_blur2)
                
                cur_corr_adv[1], cur_xent_adv[1] = sess3.run(
                                              [model3.num_correct,model3.xent],
                                              feed_dict = dict_adv2)
                
                #Third autoencoder dictionary
                cur_corr_blur[2], cur_xent_blur[2] = sess3.run(
                                              [model3.num_correct,model3.xent],
                                              feed_dict = dict_blur3)
                
                cur_corr_adv[2], cur_xent_adv[2] = sess3.run(
                                              [model3.num_correct,model3.xent],
                                              feed_dict = dict_adv3)
                
                #Fourth autoencoder dictionary
                cur_corr_blur[3], cur_xent_blur[3] = sess3.run(
                                              [model3.num_correct,model3.xent],
                                              feed_dict = dict_blur4)
                
                cur_corr_adv[3], cur_xent_adv[3] = sess3.run(
                                              [model3.num_correct,model3.xent],
                                              feed_dict = dict_adv4)
                
                
                
                #Natural
                total_corr_nat += cur_corr_nat
                total_corr_corr += cur_corr_corr
                total_xent_nat += cur_xent_nat
                total_xent_corr += cur_xent_corr
                
                #running accuracy
                total_corr_adv += cur_corr_adv
                total_corr_blur += cur_corr_blur
                total_xent_adv += cur_xent_adv
                total_xent_blur += cur_xent_blur
            
            
    
            #Regual images
            avg_xent_nat = total_xent_nat / num_eval_examples
            avg_xent_corr = total_xent_corr / num_eval_examples
            acc_nat = total_corr_nat / num_eval_examples
            acc_corr = total_corr_corr / num_eval_examples
            
            #Total accuracy
            acc_adv = total_corr_adv / num_eval_examples
            acc_blur = total_corr_blur / num_eval_examples
            avg_xent_adv = total_xent_adv / num_eval_examples
            avg_xent_blur = total_xent_blur / num_eval_examples
            
    #sys.stdout = sys.__stdout__
    print("No Autoencoder")
    print('natural: {:.2f}%'.format(100 * acc_nat))
    print('Corrupted: {:.2f}%'.format(100 * acc_corr))
    print('avg nat loss: {:.4f}'.format(avg_xent_nat))
    print('avg corr loss: {:.4f} \n'.format(avg_xent_corr))
    
    print("First Autoencoder")
    print('natural with blur: {:.2f}%'.format(100 * acc_blur[0]))
    print('adversarial: {:.2f}%'.format(100 * acc_adv[0]))
    print('avg nat with blur loss: {:.4f}'.format(avg_xent_blur[0]))
    print('avg adv loss: {:.4f} \n'.format(avg_xent_adv[0]))
    
    print("Second Autoencoder")
    print('natural with blur: {:.2f}%'.format(100 * acc_blur[1]))
    print('adversarial: {:.2f}%'.format(100 * acc_adv[1]))
    print('avg nat with blur loss: {:.4f}'.format(avg_xent_blur[1]))
    print('avg adv loss: {:.4f} \n'.format(avg_xent_adv[1]))
    
    print("Third Autoencoder")
    print('natural with blur: {:.2f}%'.format(100 * acc_blur[2]))
    print('adversarial: {:.2f}%'.format(100 * acc_adv[2]))
    print('avg nat with blur loss: {:.4f}'.format(avg_xent_blur[2]))
    print('avg adv loss: {:.4f} \n'.format(avg_xent_adv[2]))
    
    print("Fourth Autoencoder")
    print('natural with blur: {:.2f}%'.format(100 * acc_blur[3]))
    print('adversarial: {:.2f}%'.format(100 * acc_adv[3]))
    print('avg nat with blur loss: {:.4f}'.format(avg_xent_blur[3]))
    print('avg adv loss: {:.4f} \n'.format(avg_xent_adv[3]))

# eval loop
cur_checkpoint = tf.train.latest_checkpoint(model_dir)
evaluate_checkpoint(cur_checkpoint)
time.sleep(10)



