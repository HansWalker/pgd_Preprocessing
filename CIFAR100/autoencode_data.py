import math
import tensorflow.compat.v1 as tf;
from sargan_dep.sargan_models import SARGAN;
import numpy as np
from sargan_dep.sar_utilities import add_gaussian_noise

def get_channel1(x_blur_list,img_size,num_batches,GPU_ID):
    trained_model_pathx1 = 'trained_models/sargan_cifar100-x1-c1'
    trained_model_pathx2 = 'trained_models/sargan_cifar100-x2-c1'
    trained_model_pathx3 = 'trained_models/sargan_cifar100-x3-c1'
    trained_model_pathx4 = 'trained_models/sargan_cifar100-x4-c1'
    blur_batch1=[]
    blur_batch2=[]
    blur_batch3=[]
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(GPU_ID))
    config = tf.ConfigProto(gpu_options=gpu_options)
    gx1=tf.Graph()
    with gx1.as_default():
        with tf.Session(config=config) as sess1:
            sargan_model1=SARGAN(img_size, 64, img_channel=1)
            sargan_saver1= tf.train.Saver()    
            sargan_saver1 = tf.train.import_meta_graph(trained_model_pathx1+'/sargan_mnist.meta');
            sargan_saver1.restore(sess1,tf.train.latest_checkpoint(trained_model_pathx1));
            for ibatch in range(num_batches):
                processed_batch=sess1.run(sargan_model1.gen_img,feed_dict={sargan_model1.image: x_blur_list[ibatch], sargan_model1.cond: x_blur_list[ibatch]})
                x_blur_list[ibatch]=np.copy(processed_batch)
                blur_batch1.append(np.copy(processed_batch))
    gx2=tf.Graph()
    with gx2.as_default():
        with tf.Session(config=config) as sess2:
            sargan_model2=SARGAN(img_size, 64, img_channel=1)
            sargan_saver2= tf.train.Saver()    
            sargan_saver2 = tf.train.import_meta_graph(trained_model_pathx2+'/sargan_mnist.meta');
            sargan_saver2.restore(sess2,tf.train.latest_checkpoint(trained_model_pathx2));
            for ibatch in range(num_batches):
                processed_batch=sess2.run(sargan_model2.gen_img,feed_dict={sargan_model2.image: x_blur_list[ibatch], sargan_model2.cond: x_blur_list[ibatch]})
                x_blur_list[ibatch]=np.copy(processed_batch)
                blur_batch2.append(np.copy(processed_batch))
    gx3=tf.Graph()
    with gx3.as_default():
        with tf.Session(config=config) as sess3:
            sargan_model3=SARGAN(img_size, 64, img_channel=1)
            sargan_saver3= tf.train.Saver()    
            sargan_saver3 = tf.train.import_meta_graph(trained_model_pathx3+'/sargan_mnist.meta');
            sargan_saver3.restore(sess3,tf.train.latest_checkpoint(trained_model_pathx3));
            for ibatch in range(num_batches):
                processed_batch=sess3.run(sargan_model3.gen_img,feed_dict={sargan_model3.image: x_blur_list[ibatch], sargan_model3.cond: x_blur_list[ibatch]})
                x_blur_list[ibatch]=np.copy(processed_batch)
                blur_batch3.append(np.copy(processed_batch))
    gx4=tf.Graph()
    with gx4.as_default():
        with tf.Session(config=config) as sess4:
            sargan_model4=SARGAN(img_size, 64, img_channel=1)
            sargan_saver4= tf.train.Saver()    
            sargan_saver4 = tf.train.import_meta_graph(trained_model_pathx4+'/sargan_mnist.meta');
            sargan_saver4.restore(sess4,tf.train.latest_checkpoint(trained_model_pathx4));
            for ibatch in range(num_batches):
                processed_batch=sess4.run(sargan_model4.gen_img,feed_dict={sargan_model4.image: x_blur_list[ibatch], sargan_model4.cond: x_blur_list[ibatch]})
                x_blur_list[ibatch]=np.copy(processed_batch)
    return blur_batch1,blur_batch2,blur_batch3,x_blur_list
def get_channel2(x_blur_list,img_size,num_batches,GPU_ID):
    trained_model_pathx1 = 'trained_models/sargan_cifar100-x1-c2'
    trained_model_pathx2 = 'trained_models/sargan_cifar100-x2-c2'
    trained_model_pathx3 = 'trained_models/sargan_cifar100-x3-c2'
    trained_model_pathx4 = 'trained_models/sargan_cifar100-x4-c2'
    blur_batch1=[]
    blur_batch2=[]
    blur_batch3=[]
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(GPU_ID))
    config = tf.ConfigProto(gpu_options=gpu_options)
    gx1=tf.Graph()
    with gx1.as_default():
        with tf.Session(config=config) as sess1:
            sargan_model1=SARGAN(img_size, 64, img_channel=1)
            sargan_saver1= tf.train.Saver()    
            sargan_saver1 = tf.train.import_meta_graph(trained_model_pathx1+'/sargan_mnist.meta');
            sargan_saver1.restore(sess1,tf.train.latest_checkpoint(trained_model_pathx1));
            for ibatch in range(num_batches):
                processed_batch=sess1.run(sargan_model1.gen_img,feed_dict={sargan_model1.image: x_blur_list[ibatch], sargan_model1.cond: x_blur_list[ibatch]})
                x_blur_list[ibatch]=np.copy(processed_batch)
                blur_batch1.append(np.copy(processed_batch))
    gx2=tf.Graph()
    with gx2.as_default():
        with tf.Session(config=config) as sess2:
            sargan_model2=SARGAN(img_size, 64, img_channel=1)
            sargan_saver2= tf.train.Saver()    
            sargan_saver2 = tf.train.import_meta_graph(trained_model_pathx2+'/sargan_mnist.meta');
            sargan_saver2.restore(sess2,tf.train.latest_checkpoint(trained_model_pathx2));
            for ibatch in range(num_batches):
                processed_batch=sess2.run(sargan_model2.gen_img,feed_dict={sargan_model2.image: x_blur_list[ibatch], sargan_model2.cond: x_blur_list[ibatch]})
                x_blur_list[ibatch]=np.copy(processed_batch)
                blur_batch2.append(np.copy(processed_batch))
    gx3=tf.Graph()
    with gx3.as_default():
        with tf.Session(config=config) as sess3:
            sargan_model3=SARGAN(img_size, 64, img_channel=1)
            sargan_saver3= tf.train.Saver()    
            sargan_saver3 = tf.train.import_meta_graph(trained_model_pathx3+'/sargan_mnist.meta');
            sargan_saver3.restore(sess3,tf.train.latest_checkpoint(trained_model_pathx3));
            for ibatch in range(num_batches):
                processed_batch=sess3.run(sargan_model3.gen_img,feed_dict={sargan_model3.image: x_blur_list[ibatch], sargan_model3.cond: x_blur_list[ibatch]})
                x_blur_list[ibatch]=np.copy(processed_batch)
                blur_batch3.append(np.copy(processed_batch))
    gx4=tf.Graph()
    with gx4.as_default():
        with tf.Session(config=config) as sess4:
            sargan_model4=SARGAN(img_size, 64, img_channel=1)
            sargan_saver4= tf.train.Saver()    
            sargan_saver4 = tf.train.import_meta_graph(trained_model_pathx4+'/sargan_mnist.meta');
            sargan_saver4.restore(sess4,tf.train.latest_checkpoint(trained_model_pathx4));
            for ibatch in range(num_batches):
                processed_batch=sess4.run(sargan_model4.gen_img,feed_dict={sargan_model4.image: x_blur_list[ibatch], sargan_model4.cond: x_blur_list[ibatch]})
                x_blur_list[ibatch]=np.copy(processed_batch)
    return blur_batch1,blur_batch2,blur_batch3,x_blur_list
def get_channel3(x_blur_list,img_size,num_batches,GPU_ID):
    trained_model_pathx1 = 'trained_models/sargan_cifar100-x1-c3'
    trained_model_pathx2 = 'trained_models/sargan_cifar100-x2-c3'
    trained_model_pathx3 = 'trained_models/sargan_cifar100-x3-c3'
    trained_model_pathx4 = 'trained_models/sargan_cifar100-x4-c3'
    blur_batch1=[]
    blur_batch2=[]
    blur_batch3=[]
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(GPU_ID))
    config = tf.ConfigProto(gpu_options=gpu_options)
    gx1=tf.Graph()
    with gx1.as_default():
        with tf.Session(config=config) as sess1:
            sargan_model1=SARGAN(img_size, 64, img_channel=1)
            sargan_saver1= tf.train.Saver()    
            sargan_saver1 = tf.train.import_meta_graph(trained_model_pathx1+'/sargan_mnist.meta');
            sargan_saver1.restore(sess1,tf.train.latest_checkpoint(trained_model_pathx1));
            for ibatch in range(num_batches):
                processed_batch=sess1.run(sargan_model1.gen_img,feed_dict={sargan_model1.image: x_blur_list[ibatch], sargan_model1.cond: x_blur_list[ibatch]})
                x_blur_list[ibatch]=np.copy(processed_batch)
                blur_batch1.append(np.copy(processed_batch))
    gx2=tf.Graph()
    with gx2.as_default():
        with tf.Session(config=config) as sess2:
            sargan_model2=SARGAN(img_size, 64, img_channel=1)
            sargan_saver2= tf.train.Saver()    
            sargan_saver2 = tf.train.import_meta_graph(trained_model_pathx2+'/sargan_mnist.meta');
            sargan_saver2.restore(sess2,tf.train.latest_checkpoint(trained_model_pathx2));
            for ibatch in range(num_batches):
                processed_batch=sess2.run(sargan_model2.gen_img,feed_dict={sargan_model2.image: x_blur_list[ibatch], sargan_model2.cond: x_blur_list[ibatch]})
                x_blur_list[ibatch]=np.copy(processed_batch)
                blur_batch2.append(np.copy(processed_batch))
    gx3=tf.Graph()
    with gx3.as_default():
        with tf.Session(config=config) as sess3:
            sargan_model3=SARGAN(img_size, 64, img_channel=1)
            sargan_saver3= tf.train.Saver()    
            sargan_saver3 = tf.train.import_meta_graph(trained_model_pathx3+'/sargan_mnist.meta');
            sargan_saver3.restore(sess3,tf.train.latest_checkpoint(trained_model_pathx3));
            for ibatch in range(num_batches):
                processed_batch=sess3.run(sargan_model3.gen_img,feed_dict={sargan_model3.image: x_blur_list[ibatch], sargan_model3.cond: x_blur_list[ibatch]})
                x_blur_list[ibatch]=np.copy(processed_batch)
                blur_batch3.append(np.copy(processed_batch))
    gx4=tf.Graph()
    with gx4.as_default():
        with tf.Session(config=config) as sess4:
            sargan_model4=SARGAN(img_size, 64, img_channel=1)
            sargan_saver4= tf.train.Saver()    
            sargan_saver4 = tf.train.import_meta_graph(trained_model_pathx4+'/sargan_mnist.meta');
            sargan_saver4.restore(sess4,tf.train.latest_checkpoint(trained_model_pathx4));
            for ibatch in range(num_batches):
                processed_batch=sess4.run(sargan_model4.gen_img,feed_dict={sargan_model4.image: x_blur_list[ibatch], sargan_model4.cond: x_blur_list[ibatch]})
                x_blur_list[ibatch]=np.copy(processed_batch)
    return blur_batch1,blur_batch2,blur_batch3,x_blur_list
    
def get_processed_data(x_batch,img_size,GPU_ID):
    NOISE_STD_RANGE = [0.0, 0.02]
    x_batch=np.expand_dims(add_gaussian_noise(x_batch, sd=NOISE_STD_RANGE[1]),4)
    channel1=[]
    channel2=[]
    channel3=[]
    num_batches=math.floor(len(x_batch)/64)
    differnce=len(x_batch)-num_batches*64
    num_batches2=num_batches
    batch1=np.zeros([64,img_size[0],img_size[1],img_size[2]])
    for i in range(num_batches):
        channel1.append(np.copy(x_batch[(64*i):(64*i+64),:,:,0]))
        channel2.append(np.copy(x_batch[(64*i):(64*i+64),:,:,1]))
        channel3.append(np.copy(x_batch[(64*i):(64*i+64),:,:,2]))
        
    if(differnce!=0):
        batch1[0:differnce,:,:,:]=x_batch[(64*num_batches):(64*num_batches+differnce),:,:,0]
        channel1.append(np.copy(batch1))
        batch1[0:differnce,:,:,:]=x_batch[(64*num_batches):(64*num_batches+differnce),:,:,1]
        channel2.append(np.copy(batch1))
        batch1[0:differnce,:,:,:]=x_batch[(64*num_batches):(64*num_batches+differnce),:,:,2]
        channel3.append(np.copy(batch1))
        num_batches2+=1
    blur_batch11,blur_batch21,blur_batch31,blur_batch41=get_channel1(channel1, img_size,num_batches2,GPU_ID)
    blur_batch12,blur_batch22,blur_batch32,blur_batch42=get_channel2(channel2, img_size,num_batches2,GPU_ID)
    blur_batch13,blur_batch23,blur_batch33,blur_batch43=get_channel3(channel3, img_size,num_batches2,GPU_ID)
    next_batch1=np.zeros([len(x_batch),img_size[0],img_size[1],3])
    next_batch2=np.zeros([len(x_batch),img_size[0],img_size[1],3])
    next_batch3=np.zeros([len(x_batch),img_size[0],img_size[1],3])
    next_batch4=np.zeros([len(x_batch),img_size[0],img_size[1],3])
    for i in range(num_batches):
        next_batch1[(64*i):(64*i+64),:,:,0]=np.copy(blur_batch11[i][:,:,:,0])
        next_batch1[(64*i):(64*i+64),:,:,1]=np.copy(blur_batch12[i][:,:,:,0])
        next_batch1[(64*i):(64*i+64),:,:,2]=np.copy(blur_batch13[i][:,:,:,0])
        
        next_batch2[(64*i):(64*i+64),:,:,0]=np.copy(blur_batch21[i][:,:,:,0])
        next_batch2[(64*i):(64*i+64),:,:,1]=np.copy(blur_batch22[i][:,:,:,0])
        next_batch2[(64*i):(64*i+64),:,:,2]=np.copy(blur_batch23[i][:,:,:,0])
        
        next_batch3[(64*i):(64*i+64),:,:,0]=np.copy(blur_batch31[i][:,:,:,0])
        next_batch3[(64*i):(64*i+64),:,:,1]=np.copy(blur_batch32[i][:,:,:,0])
        next_batch3[(64*i):(64*i+64),:,:,2]=np.copy(blur_batch33[i][:,:,:,0])
        
        next_batch4[(64*i):(64*i+64),:,:,0]=np.copy(blur_batch41[i][:,:,:,0])
        next_batch4[(64*i):(64*i+64),:,:,1]=np.copy(blur_batch42[i][:,:,:,0])
        next_batch4[(64*i):(64*i+64),:,:,2]=np.copy(blur_batch43[i][:,:,:,0])
    if(differnce!=0):
        next_batch1[(64*num_batches):(64*num_batches+differnce),:,:,0]=np.copy(blur_batch11[num_batches][0:differnce,:,:,0])
        next_batch1[(64*num_batches):(64*num_batches+differnce),:,:,1]=np.copy(blur_batch12[num_batches][0:differnce,:,:,0])
        next_batch1[(64*num_batches):(64*num_batches+differnce),:,:,2]=np.copy(blur_batch13[num_batches][0:differnce,:,:,0])
        
        next_batch2[(64*num_batches):(64*num_batches+differnce),:,:,0]=np.copy(blur_batch21[num_batches][0:differnce,:,:,0])
        next_batch2[(64*num_batches):(64*num_batches+differnce),:,:,1]=np.copy(blur_batch22[num_batches][0:differnce,:,:,0])
        next_batch2[(64*num_batches):(64*num_batches+differnce),:,:,2]=np.copy(blur_batch23[num_batches][0:differnce,:,:,0])
        
        next_batch3[(64*num_batches):(64*num_batches+differnce),:,:,0]=np.copy(blur_batch31[num_batches][0:differnce,:,:,0])
        next_batch3[(64*num_batches):(64*num_batches+differnce),:,:,1]=np.copy(blur_batch32[num_batches][0:differnce,:,:,0])
        next_batch3[(64*num_batches):(64*num_batches+differnce),:,:,2]=np.copy(blur_batch33[num_batches][0:differnce,:,:,0])
        
        next_batch4[(64*num_batches):(64*num_batches+differnce),:,:,0]=np.copy(blur_batch41[num_batches][0:differnce,:,:,0])
        next_batch4[(64*num_batches):(64*num_batches+differnce),:,:,1]=np.copy(blur_batch42[num_batches][0:differnce,:,:,0])
        next_batch4[(64*num_batches):(64*num_batches+differnce),:,:,2]=np.copy(blur_batch43[num_batches][0:differnce,:,:,0])
    return next_batch1,next_batch2,next_batch3,next_batch4