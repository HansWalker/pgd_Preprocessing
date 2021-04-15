from tensorflow.keras.datasets import cifar100;
import tensorflow.keras as keras;
import tensorflow as tf;
from PIL import Image
import numpy as np;
from autoencode_data import get_processed_data
import math
img_size1 = [32,32,3]
img_size2 = [32,32,1]
GPU_ID = 0
(x_train, y_train), (x_test, y_test) = cifar100.load_data();
def preprocess_data(X,Y,model,GPU_ID):
    X=np.array(keras.applications.resnet.preprocess_input(X))
    x_test_o=np.copy(X)
    loss = keras.losses.CategoricalCrossentropy()
    epsilon=.005*255
    pert=.0005*255
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
    first,second,third,fourth=get_processed_data(X_p, img_size2,GPU_ID)
    
    first=np.array(first)
    first=(first-np.min(first))/(np.max(first)-np.min(first))
    first[:,:,:,0]=(first[:,:,:,0]*255-103.939)
    first[:,:,:,1]=(first[:,:,:,1]*255-116.779)
    first[:,:,:,2]=(first[:,:,:,2]*255-123.68)
    
    
    second=np.array(second)
    second=(third-np.min(second))/(np.max(second)-np.min(second))
    second[:,:,:,0]=(second[:,:,:,0]*255-103.939)
    second[:,:,:,1]=(second[:,:,:,1]*255-116.779)
    second[:,:,:,2]=(second[:,:,:,2]*255-123.68)
    
    third=np.array(third)
    third=(third-np.min(third))/(np.max(third)-np.min(third))
    third[:,:,:,0]=(third[:,:,:,0]*255-103.939)
    third[:,:,:,1]=(third[:,:,:,1]*255-116.779)
    third[:,:,:,2]=(third[:,:,:,2]*255-123.68)
    
    fourth=np.array(fourth)
    fourth[:,:,:,0]=(fourth[:,:,:,0]-np.min(fourth[:,:,:,0]))/(np.max(fourth[:,:,:,0])-np.min(fourth[:,:,:,0]))
    fourth[:,:,:,1]=(fourth[:,:,:,1]-np.min(fourth[:,:,:,1]))/(np.max(fourth[:,:,:,1])-np.min(fourth[:,:,:,1]))
    fourth[:,:,:,2]=(fourth[:,:,:,2]-np.min(fourth[:,:,:,2]))/(np.max(fourth[:,:,:,2])-np.min(fourth[:,:,:,2]))

    fourth[:,:,:,0]=(fourth[:,:,:,0]*255-103.939)
    fourth[:,:,:,1]=(fourth[:,:,:,1]*255-116.779)
    fourth[:,:,:,2]=(fourth[:,:,:,2]*255-123.68)
    
    X[:,:,:,0]=tf.math.add(X[:,:,:,0]*255,-103.939)
    X[:,:,:,1]=tf.math.add(X[:,:,:,1]*255,-116.779)
    X[:,:,:,2]=tf.math.add(X[:,:,:,2]*255,-123.68)
    # encode to one-hot
    return x_test_o,X,first,second,third,fourth
CALLBACKS = []
MODEL_PATH = 'pgd/models'
model_name='/cifar100'
inputs = keras.Input(shape=(32, 32, 3))
upscale = keras.layers.Lambda(lambda x: tf.compat.v1.image.resize_image_with_pad(x,
                              160,
                              160,
                              method=tf.image.ResizeMethod.BILINEAR))(inputs)
model=keras.models.load_model(MODEL_PATH+model_name)


y_test = keras.utils.to_categorical(y_test, 100)
x_test_o,X,first,second,third,fourth= preprocess_data(x_test,y_test,model,GPU_ID)
psnr_value=np.array(tf.image.psnr(x_test_o,fourth,max_val=255))
psnr=0
for i in range(len(psnr_value)):
    psnr+=psnr_value[i]
psnr/=len(psnr_value)
print("\n\n\nDONE\n\n\n")
# upscale layer
eval_results_o=model.evaluate(x_test_o, y_test, batch_size=64, verbose=0)
eval_results_adv=model.evaluate(X, y_test, batch_size=64, verbose=0)
eval_results_1=model.evaluate(first, y_test, batch_size=64, verbose=0)
eval_results_2=model.evaluate(second, y_test, batch_size=64, verbose=0)
eval_results_3=model.evaluate(third, y_test, batch_size=64, verbose=0)
eval_results_4=model.evaluate(fourth, y_test, batch_size=64, verbose=0)
print("PSNR value= ",psnr,"\n")
print("Natural Image Accuracy:\n",eval_results_o[1],"\n")
print("Adversarial Image Accuracy:\n",eval_results_adv[1],"\n")
print("First Autoenocoded Image Accuracy:\n",eval_results_1[1],"\n")
print("Second Autoenocoded Image Accuracy:\n",eval_results_2[1],"\n")
print("Third Autoenocoded Image Accuracy:\n",eval_results_3[1],"\n")
print("Fourth Autoenocoded Image Accuracy:\n",eval_results_4[1],"\n")