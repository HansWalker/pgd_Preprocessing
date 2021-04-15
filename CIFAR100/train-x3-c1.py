import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sargan_dep.sargan_models import SARGAN
from tqdm import tqdm
import time
import numpy as np
from sargan_dep.sar_utilities import add_gaussian_noise
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
tf.enable_eager_execution

img_size = [32,32,1]
img_size2 = [64,32,32]
experiment_name = '/sargan_cifar100'
trained_model_path1 = 'trained_models/sargan_cifar100-x1-c1'
trained_model_path2 = 'trained_models/sargan_cifar100-x2-c1'
trained_model_path = 'trained_models/sargan_cifar100-x3-c1'
data_root='sar_data/cifar100'
NUM_ITERATION = 85
BATCH_SIZE = 64
GPU_ID = 0
MAX_EPOCH = 3000
LEARNING_RATE = 0.001
SAVE_EVERY_EPOCH = 10

#continuing training from a previous model
retrain=0
####
#GETTING IMAGES
####
NOISE_STD_RANGE = [0.0, 0.12]

def get_data(train_batch_size):
    
    data_transform = Compose([ToTensor()])
    
    train_loader = DataLoader(datasets.CIFAR100(root=data_root, train=True, transform=data_transform, target_transform=None, download=True),
                              batch_size=train_batch_size, shuffle=True)
    
    
    return train_loader

def transfrom_data(NUM_TEST_PER_EPOCH):
    encoded_data=[]
    original_data=[]
    train_loader= get_data(BATCH_SIZE)
    trainiter = iter(train_loader)
    for i in range(NUM_ITERATION):
        features2, labels = next(trainiter)
        features2 = np.array(tf.keras.applications.resnet.preprocess_input(features2.data.numpy().transpose(0,2,3,1)*255))
        features=np.zeros([len(features2),img_size[0],img_size[1],img_size[2]])
        features[:,:,:,0]=(features2[:,:,:,0])
        features=np.clip(features+103.939,0,255)/255
        original_images=np.copy(features)
        for k in range(len(features)):
            features[k]=add_gaussian_noise(features[k], sd=np.random.uniform(NOISE_STD_RANGE[0], NOISE_STD_RANGE[1]))
        encoded_data.append(np.copy(features))
        original_data.append(np.copy(original_images))
                
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(GPU_ID))
    config = tf.ConfigProto(gpu_options=gpu_options)
    gx1 = tf.Graph()
    with gx1.as_default():
        with tf.Session(config=config) as sess1:
             sargan_model1=SARGAN(img_size, BATCH_SIZE, img_channel=1)
             sargan_saver1= tf.train.Saver()    
             sargan_saver1 = tf.train.import_meta_graph(trained_model_path1+'/sargan_mnist.meta');
             sargan_saver1.restore(sess1,tf.train.latest_checkpoint(trained_model_path1));
             for ibatch in range(NUM_TEST_PER_EPOCH):
                 processed_batch=sess1.run(sargan_model1.gen_img,feed_dict={sargan_model1.image: encoded_data[ibatch], sargan_model1.cond: encoded_data[ibatch]})
                 encoded_data[ibatch]=(np.copy(processed_batch))
    gx2 = tf.Graph()
    with gx2.as_default():
        with tf.Session(config=config) as sess2:
             sargan_model2=SARGAN(img_size, BATCH_SIZE, img_channel=1)
             sargan_saver2= tf.train.Saver()    
             sargan_saver2 = tf.train.import_meta_graph(trained_model_path2+'/sargan_mnist.meta');
             sargan_saver2.restore(sess2,tf.train.latest_checkpoint(trained_model_path2));
             for ibatch in range(NUM_TEST_PER_EPOCH):
                 processed_batch=sess2.run(sargan_model2.gen_img,feed_dict={sargan_model2.image: encoded_data[ibatch], sargan_model2.cond: encoded_data[ibatch]})
                 encoded_data[ibatch]=(np.copy(processed_batch))


    return encoded_data, original_data

def main(args):
    model = SARGAN(img_size, BATCH_SIZE, img_channel=img_size[2])
    with tf.variable_scope("d_opt",reuse=tf.AUTO_REUSE):
        d_opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.d_loss, var_list=model.d_vars)
    with tf.variable_scope("g_opt",reuse=tf.AUTO_REUSE):
        g_opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.g_loss, var_list=model.g_vars)
    if(retrain==0):
        saver = tf.train.Saver(max_to_keep=20)
    else:
        saver = tf.train.import_meta_graph(trained_model_path+'/sargan_mnist.meta');
    
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(GPU_ID))
    config = tf.ConfigProto(gpu_options=gpu_options)
    
    progress_bar = tqdm(range(MAX_EPOCH), unit="epoch")
    #list of loss values each item is the loss value of one ieteration
    train_d_loss_values = []
    train_g_loss_values = []
    
    
    #test_imgs, test_classes = get_data(test_filename)
    #imgs, classes = get_data(train_filename)
    with tf.Session(config=config) as sess:
        if(retrain==0):
            sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess,tf.train.latest_checkpoint(trained_model_path));
        #test_copies = test_imgs.astype('float32')
        for epoch in progress_bar:
            NUM_TEST_PER_EPOCH = 1
            counter = 0
            epoch_start_time = time.time()
            encoded_data, original_data=transfrom_data(NUM_TEST_PER_EPOCH)
            #shuffle(copies)
            #divide the images into equal sized batches
            #image_batches = np.array(list(chunks(copies, BATCH_SIZE)))
            for i in range (NUM_ITERATION):
                #getting a batch from the training data
                #one_batch_of_imgs = image_batches[i]                
                #copy the batch
                features=original_data[i]
                #corrupt the images
                corrupted_batch = encoded_data[i]
                _, m = sess.run([d_opt, model.d_loss], feed_dict={model.image:features, model.cond:corrupted_batch})
                _, M = sess.run([g_opt, model.g_loss], feed_dict={model.image:features, model.cond:corrupted_batch})
                train_d_loss_values.append(m)
                train_g_loss_values.append(M)
                #print some notifications
                counter += 1
                if counter % 25 == 0:
                    print("\rEpoch [%d], Iteration [%d]: time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, counter, time.time() - epoch_start_time, m, M))
                
            # save the trained network
            if epoch % SAVE_EVERY_EPOCH == 0:
                save_path = saver.save(sess, (trained_model_path+"/sargan_mnist"))
                print("\n\nModel saved in file: %s\n\n" % save_path)       
        
if __name__ == '__main__':
    main([])
