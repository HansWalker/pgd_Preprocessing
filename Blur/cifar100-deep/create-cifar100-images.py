"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import sys
import time

import tensorflow as tf

from pgd.cifar100.model import Model
from pgd.cifar100.pgd_attack import LinfPGDAttack
from torchvision import datasets
from torch.utils.data import DataLoader

from sargan_dep.sargan_models import SARGAN
from sargan_dep.sar_utilities import add_gaussian_noise
import numpy as np
from torchvision.transforms import Compose, ToTensor
from PIL import Image
data_root='sar_data/cifar100'


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
def get_data(train_batch_size):
    
    data_transform = Compose([ToTensor()])
    
    train_loader = DataLoader(datasets.CIFAR100(root=data_root, train=True, transform=data_transform, target_transform=None, download=True),
                              batch_size=train_batch_size, shuffle=True)
    
    
    return train_loader
global_step = tf.contrib.framework
# Global constants
with open('pgd/cifar100/config_cifar100.json') as config_file:
  config = json.load(config_file)
num_eval_examples = 64
eval_batch_size = 64
eval_on_cpu = config['eval_on_cpu']

model_dir = config['model_dir-n']
# Set upd the data, hyperparameters, and the model


img_size = [32,32,1]
#Autoencoder models
trained_model_path = 'trained_models/sargan_cifar100-x1-1'
trained_model_path2 = 'trained_models/sargan_cifar100-x2-1'
trained_model_path3 = 'trained_models/sargan_cifar100-x3-1'
trained_model_path4 = 'trained_models/sargan_cifar100-x4-1'
BATCH_SIZE = 64
NOISE_STD_RANGE = [0.1, 0.1]

if eval_on_cpu:
  with tf.device("/cpu:0"):
      model = Model()
      attack = LinfPGDAttack(model, 
                           .015,#config['epsilon'],
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
    with tf.Session() as sess:
    # Restore the checkpoint
        saver.restore(sess, filename);
    
        # Iterate over the samples batch-by-batch
        #number of batches
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        x_blur_list4=[]
        x_adv_list4=[]
        #Storing y values
        
        train_loader= get_data(BATCH_SIZE)
        trainiter = iter(cycle(train_loader))
        for ibatch in range(num_batches):
                
            x_batch2, y_batch = next(trainiter)
            x_batch2 = np.array(x_batch2.data.numpy().transpose(0,2,3,1))*255
            x_batch=np.zeros([len(x_batch2),img_size[0]*img_size[1]])
            
            
            
            for i in range(len(x_batch2)):
                nextimage=Image.fromarray((x_batch2[i]).astype(np.uint8))
                nextimage=nextimage.convert('L')
                x_batch[i]=np.array(nextimage,dtype='float32').reshape([img_size[0]*img_size[1]])/255
                
            x_batch3=np.zeros([img_size[0],img_size[1]])
            x_batch3[:,:]=(x_batch[0].reshape([img_size[0],img_size[1]]))[:,:]*255
            nextimage=Image.fromarray(x_batch3.astype(np.uint8))
            nextimage.save("CIFAR100_NATURALX0.Jpg","JPEG")
            x_batch_adv = attack.perturb(x_batch, y_batch, sess)
            
            x_batch2=np.zeros([len(x_batch),img_size[0],img_size[1],img_size[2]])
            x_batch_adv2=np.zeros([len(x_batch),img_size[0],img_size[1],img_size[2]])
            for k in range(len(x_batch)):
                x_batch2[k]=add_gaussian_noise(x_batch[k].reshape([img_size[0],img_size[1],img_size[2]]), sd=np.random.uniform(NOISE_STD_RANGE[1], NOISE_STD_RANGE[1]))
                x_batch_adv2[k]=add_gaussian_noise(x_batch_adv[k].reshape([img_size[0],img_size[1],img_size[2]]), sd=np.random.uniform(NOISE_STD_RANGE[1], NOISE_STD_RANGE[1]))
            
            x_blur_list4.append(x_batch2)
            x_adv_list4.append(x_batch_adv2)
            
            x_batch3=np.zeros([img_size[0],img_size[1]])
            x_batch3[:,:]=x_blur_list4[0][0,:,:,0]*255
            nextimage=Image.fromarray(x_batch3.astype(np.uint8))
            nextimage.save("CIFAR100_BLURX0.Jpg","JPEG")
            x_batch3=np.zeros([img_size[0],img_size[1]])
            x_batch3[:,:]=x_adv_list4[0][0,:,:,0]*255
            nextimage=Image.fromarray(x_batch3.astype(np.uint8))
            nextimage.save("CIFAR100_ADVX0.Jpg","JPEG")
            
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
                
            x_batch3[:,:]=x_blur_list4[0][0,:,:,0]*255
            nextimage=Image.fromarray(x_batch3.astype(np.uint8))
            nextimage.save("CIFAR100_BLURX1.Jpg","JPEG")
            x_batch3=np.zeros([img_size[0],img_size[1]])
            x_batch3[:,:]=x_adv_list4[0][0,:,:,0]*255
            nextimage=Image.fromarray(x_batch3.astype(np.uint8))
            nextimage.save("CIFAR100_ADVX1.Jpg","JPEG")
            
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
            x_batch3=np.zeros([img_size[0],img_size[1]])
            x_batch3[:,:]=x_blur_list4[0][0,:,:,0]*255
            nextimage=Image.fromarray(x_batch3.astype(np.uint8))
            nextimage.save("CIFAR100_BLURX2.Jpg","JPEG")
            x_batch3=np.zeros([img_size[0],img_size[1]])
            x_batch3[:,:]=x_adv_list4[0][0,:,:,0]*255
            nextimage=Image.fromarray(x_batch3.astype(np.uint8))
            nextimage.save("CIFAR100_ADVX2.Jpg","JPEG")
                
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
                
            x_batch3=np.zeros([img_size[0],img_size[1]])
            x_batch3[:,:]=x_blur_list4[0][0,:,:,0]*255
            nextimage=Image.fromarray(x_batch3.astype(np.uint8))
            nextimage.save("CIFAR100_BLURX3.Jpg","JPEG")
            x_batch3=np.zeros([img_size[0],img_size[1]])
            x_batch3[:,:]=x_adv_list4[0][0,:,:,0]*255
            nextimage=Image.fromarray(x_batch3.astype(np.uint8))
            nextimage.save("CIFAR100_ADVX3.Jpg","JPEG")
            
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
                
                x_batch3=np.zeros([img_size[0],img_size[1]])
                x_batch3[:,:]=processed_batch[0,:,:,0]*255
                nextimage=Image.fromarray(x_batch3.astype(np.uint8))
                nextimage.save("CIFAR100_ADVX4.Jpg","JPEG")
                
                blurred_batch=sessx4.run(sargan_model4.gen_img,feed_dict={sargan_model4.image: x_blur_list4[ibatch], sargan_model4.cond: x_blur_list4[ibatch]})
                x_blur_list4[ibatch]=blurred_batch.reshape([len(x_batch),img_size[0]*img_size[1]])
                
                x_batch3=np.zeros([img_size[0],img_size[1]])
                x_batch3[:,:]=blurred_batch[0,:,:,0]*255
                nextimage=Image.fromarray(x_batch3.astype(np.uint8))
                nextimage.save("CIFAR100_BLURX4.Jpg","JPEG")


# eval loop
cur_checkpoint = tf.train.latest_checkpoint(model_dir)
evaluate_checkpoint(cur_checkpoint)
time.sleep(10)



