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
from sargan_dep.sar_utilities import add_gaussian_noise
from sargan_dep.sargan_models import SARGAN
import tensorflow as tf
import numpy as np
from torchvision import  datasets
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader

from pgd.cifar10at32.model import Model
from pgd.cifar10at32.pgd_attack import LinfPGDAttack

from PIL import Image
trained_model_path1 = 'trained_models/sargan_cifar10-x1-4' 
trained_model_path2 = 'trained_models/sargan_cifar10-x2-4'
trained_model_path3 = 'trained_models/sargan_cifar10-x3-4'
trained_model_path4 = 'trained_models/sargan_cifar10-x4-4'
with open('pgd/cifar10at32/config_cifar10.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']
# Setting up the Tensorboard and checkpoint outputs
model_dir = "pgd/models/cifar10-n-auto-2"
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
# Setting up the data and the model
img_size = [32,32,1]
data_root='sar_data/cifar10'

def get_data(train_batch_size):
    data_transform = Compose([ToTensor()])
    
    train_loader = DataLoader(datasets.CIFAR10(root=data_root, train=True, transform=data_transform, target_transform=None, download=True),
                              batch_size=train_batch_size, shuffle=True)
    
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
def transform_data(num_batches,BATCH_SIZE):
    g1=tf.Graph()
    g2=tf.Graph()
    g3=tf.Graph()
    g4=tf.Graph()
    encoded_data=[]
    labels_list=[]
    original_data=[]
    train_loader= get_data(BATCH_SIZE)
    trainiter = iter(cycle(train_loader))
    
    with g1.as_default():
        with tf.Session() as sess1:
            sargan_model1=SARGAN(img_size, BATCH_SIZE, img_channel=1)
            sargan_saver= tf.train.Saver()    
            sargan_saver = tf.train.import_meta_graph(trained_model_path1+'/sargan_mnist.meta');
            sargan_saver.restore(sess1,tf.train.latest_checkpoint(trained_model_path1));
            for i in range(num_batches):
                features, labels = next(trainiter)
                features = features.data.numpy().transpose(0,2,3,1)*255
                features2=np.zeros([len(features),img_size[0],img_size[1],1])
                for j in range(BATCH_SIZE):
                    nextimage=Image.fromarray((features[j]).astype(np.uint8))
                    nextimage=nextimage.convert('L')
                    features2[j,:,:,0]=np.array(nextimage,dtype='float32')/255
                original_data.append(features2.reshape([BATCH_SIZE,img_size[0]*img_size[1]]))
                processed_batch=sess1.run(sargan_model1.gen_img,feed_dict={sargan_model1.image: features2, sargan_model1.cond: features2})
                encoded_data.append(processed_batch)
                labels_list.append(labels)
    with g2.as_default():
        with tf.Session() as sess2:
            sargan_model2=SARGAN(img_size, BATCH_SIZE, img_channel=1)
            sargan_saver2= tf.train.Saver()    
            sargan_saver2 = tf.train.import_meta_graph(trained_model_path2+'/sargan_mnist.meta');
            sargan_saver2.restore(sess2,tf.train.latest_checkpoint(trained_model_path2));
            for i in range(num_batches):
                encoded_data[i]=sess2.run(sargan_model2.gen_img,feed_dict={sargan_model2.image: encoded_data[i], sargan_model2.cond: encoded_data[i]})
    with g3.as_default():
        with tf.Session() as sess3:
            sargan_model3=SARGAN(img_size, BATCH_SIZE, img_channel=1)
            sargan_saver3 = tf.train.Saver()    
            sargan_saver3 = tf.train.import_meta_graph(trained_model_path3+'/sargan_mnist.meta');
            sargan_saver3.restore(sess3,tf.train.latest_checkpoint(trained_model_path3));
            for i in range(num_batches):
                encoded_data[i]=sess3.run(sargan_model3.gen_img,feed_dict={sargan_model3.image: encoded_data[i], sargan_model3.cond: encoded_data[i]})
    with g4.as_default():
        with tf.Session() as sess4:
            sargan_model4=SARGAN(img_size, BATCH_SIZE, img_channel=1)
            sargan_saver4= tf.train.Saver()    
            sargan_saver4 = tf.train.import_meta_graph(trained_model_path4+'/sargan_mnist.meta');
            sargan_saver4.restore(sess4,tf.train.latest_checkpoint(trained_model_path4));
            for i in range(num_batches):
                encoded_data[i]=sess4.run(sargan_model4.gen_img,feed_dict={sargan_model4.image: encoded_data[i], sargan_model4.cond: encoded_data[i]}).reshape([BATCH_SIZE,img_size[0]*img_size[1]])

    return encoded_data, labels_list
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model()

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent,
                                                   global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])


# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_image)
merged_summaries = tf.summary.merge_all()

shutil.copy('pgd/cifar10/config_cifar10.json', model_dir)


with tf.Session() as sess:
    # Initialize the summary writer, global variables, and our time counter.
    #saver.restore(sess,tf.train.latest_checkpoint(model_dir))
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    training_time = 0.0
    # Main training loop
    for ii in range(int(max_num_training_steps/(num_checkpoint_steps*10))):
        allx_data, labels, allo_data=transform_data(num_checkpoint_steps*10,batch_size)
        for jj in range(num_checkpoint_steps*10):            
            # Compute Adversarial Perturbations
            start = timer()
            x_batch_adv =attack.perturb(allo_data[jj], labels[jj], sess)
            end = timer()
            training_time += end - start
            
            nat_dict = {model.x_input: allx_data[jj],
                    model.y_input: labels[jj]}
            
            adv_dict = {model.x_input: x_batch_adv,
                    model.y_input: labels[jj]}
            
                # Output to stdout
            if (ii*num_checkpoint_steps*10+jj) % num_output_steps == 0:
                nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
                adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
                print('Step {}:    ({})'.format((ii*10*num_checkpoint_steps+jj), datetime.now()))
                print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
                print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
                if (ii*10*num_checkpoint_steps+jj) != 0:
                    print('    {} examples per second'.format(
                        num_output_steps * batch_size / training_time))
                    training_time = 0.0
            # Tensorboard summaries
            #if (ii*num_output_steps+jj) % num_summary_steps == 0:
            #summary = sess.run(merged_summaries, feed_dict=adv_dict)
            #summary_writer.add_summary(summary, global_step.eval(sess))
            
            # Write a checkpoint
            if (ii*10*num_checkpoint_steps+jj) % num_checkpoint_steps == 0:
                saver.save(sess,
                           os.path.join(model_dir, 'checkpoint'),
                           global_step=global_step)
            
            # Actual training step
            start = timer()
            sess.run(train_step, feed_dict=nat_dict)
            end = timer()
            training_time += end - start