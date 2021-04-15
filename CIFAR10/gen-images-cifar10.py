from tensorflow.keras.datasets import cifar10;
import tensorflow.keras as keras;
import tensorflow as tf;
tf.compat.v1.enable_eager_execution()
from PIL import Image
import numpy as np;
from autoencode_data_gen import get_processed_data
import math
img_size1 = [32,32,3]
img_size2 = [32,32,1]
image_index=5;
(x_train, y_train), (x_test, y_test) = cifar10.load_data();
def preprocess_data(X,Y,model,image_index):
    X=np.array(keras.applications.resnet.preprocess_input(X))
    im_trans=X[image_index]
    im_trans[:,:,2]=im_trans[:,:,0]+103.939
    im_trans[:,:,1]=im_trans[:,:,1]+116.779
    im_trans[:,:,0]=im_trans[:,:,2]+123.68
    loss = keras.losses.CategoricalCrossentropy()
    epsilon=.01*255
    pert=.001*255
    for i in range(int(len(X)/100)):
        Z=tf.cast(X[i*100:(100*i+100)], tf.float32)
        Y_next=Y[i*100:(100*i+100)]
        for j in range((int)(math.floor(epsilon/pert))):
            with tf.GradientTape() as grad_tracker:
                grad_tracker.watch(Z);
                prediction=model(Z);
                next_loss=loss(Y_next,prediction);
            grad = grad_tracker.gradient(next_loss,Z);
            signed_grad = tf.sign(grad);
            Z=Z+pert*signed_grad;
            Z1 = tf.clip_by_value(Z[:,:,:,0], -103.939, 151.061);
            Z2 = tf.clip_by_value(Z[:,:,:,1], -116.779, 138.221);
            Z3 = tf.clip_by_value(Z[:,:,:,2], -123.68, 131.32);
            Z=tf.stack([Z1,Z2,Z3],axis=3);
        for k in range(100):
            X[100*i+k,:,:,0]=tf.math.add(Z[k,:,:,0],103.939)/255
            X[100*i+k,:,:,1]=tf.math.add(Z[k,:,:,1],116.779)/255
            X[100*i+k,:,:,2]=tf.math.add(Z[k,:,:,2],123.68)/255
    X_p=X
    im_corr=X_p[image_index]
    im_corr[:,:,0]=im_corr[:,:,2]
    im_corr[:,:,1]=im_corr[:,:,1]
    im_corr[:,:,2]=im_corr[:,:,0]
    
    X_p,im1,im2,im3,im4=get_processed_data(X_p, img_size2,image_index)
    
    X_p[:,:,:,0]=tf.math.add(X_p[:,:,:,0]*255,-103.939)
    X_p[:,:,:,1]=tf.math.add(X_p[:,:,:,1]*255,-116.779)
    X_p[:,:,:,2]=tf.math.add(X_p[:,:,:,2]*255,-123.68)
    
    X[:,:,:,0]=tf.math.add(X[:,:,:,0]*255,-103.939)
    X[:,:,:,1]=tf.math.add(X[:,:,:,1]*255,-116.779)
    X[:,:,:,2]=tf.math.add(X[:,:,:,2]*255,-123.68)
    # encode to one-hot
    return X_p,X, im_trans,im_corr, im1,im2,im3,im4
CALLBACKS = []
MODEL_PATH = 'pgd/models/cifar10-n-deep'
model_name='/Cifar10_Model'
optimizer = keras.optimizers.Adam()
inputs = keras.Input(shape=(32, 32, 3))
upscale = keras.layers.Lambda(lambda x: tf.compat.v1.image.resize_image_with_pad(x,
                              160,
                              160,
                              method=tf.image.ResizeMethod.BILINEAR))(inputs)
model=keras.models.load_model(MODEL_PATH+model_name)


y_test = keras.utils.to_categorical(y_test, 10)
im_orig=x_test[image_index]
x_test,X,im_trans,im_corr,im1,im2,im3,im4 = preprocess_data(x_test,y_test,model,image_index)
print("im1: ",np.min(im1),"   ",np.max(im1),"\n")
print("im2: ",np.min(im2),"   ",np.max(im2),"\n")
print("im3: ",np.min(im3),"   ",np.max(im3),"\n")
print("im4: ",np.min(im4),"   ",np.max(im4),"\n\n\n")
nextimage=Image.fromarray(im_orig.astype(np.uint8))
nextimage=nextimage.resize([320,320])
nextimage.save("Cifar10_Original.Jpg","JPEG")

nextimage=Image.fromarray(im_trans.astype(np.uint8))
nextimage=nextimage.resize([320,320])
nextimage.save("Cifar10_Transformed.Jpg","JPEG")

nextimage=Image.fromarray(im_corr.astype(np.uint8))
nextimage=nextimage.resize([320,320])
nextimage.save("Cifar10_Corrupted.Jpg","JPEG")

im1=(im1-np.min(im1))/(np.max(im1)-np.min(im1))
nextimage=Image.fromarray((im1*255).astype(np.uint8))
nextimage=nextimage.resize([320,320])
nextimage.save("Cifar10_First_autoencoder.Jpg","JPEG")

im2=(im2-np.min(im2))/(np.max(im2)-np.min(im2))
nextimage=Image.fromarray((im2*255).astype(np.uint8))
nextimage=nextimage.resize([320,320])
nextimage.save("Cifar10_Second_autoencoder.Jpg","JPEG")

im3=(im3-np.min(im3))/(np.max(im3)-np.min(im3))
nextimage=Image.fromarray((im3*255).astype(np.uint8))
nextimage=nextimage.resize([320,320])
nextimage.save("Cifar10_Third_autoencoder.Jpg","JPEG")

im4=(im4-np.min(im4))/(np.max(im4)-np.min(im4))
nextimage=Image.fromarray((im4*255).astype(np.uint8))
nextimage=nextimage.resize([320,320])
nextimage.save("Cifar10_Fourth_autoencoder.Jpg","JPEG")
