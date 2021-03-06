from tensorflow.keras.datasets import cifar10;
import tensorflow.keras as keras;
import tensorflow as tf;
tf.compat.v1.enable_eager_execution()
from PIL import Image
import numpy as np;
from autoencode_data import get_processed_data
import math
img_size1 = [32,32,3]
img_size2 = [32,32,1]
(x_train, y_train), (x_test, y_test) = cifar10.load_data();
GPU_ID = 0
def preprocess_data(X,Y,model,GPU_ID):
    X=np.array(keras.applications.resnet.preprocess_input(X))
    x_test_o=np.copy(X)
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
    first,second,third,fourth=get_processed_data(X_p, img_size2,GPU_ID)
    first[:,:,:,0]=tf.math.add(first[:,:,:,0]*255,-103.939)
    first[:,:,:,1]=tf.math.add(first[:,:,:,1]*255,-116.779)
    first[:,:,:,2]=tf.math.add(first[:,:,:,2]*255,-123.68)
    
    second[:,:,:,0]=tf.math.add(second[:,:,:,0]*255,-103.939)
    second[:,:,:,1]=tf.math.add(second[:,:,:,1]*255,-116.779)
    second[:,:,:,2]=tf.math.add(second[:,:,:,2]*255,-123.68)
    
    third[:,:,:,0]=tf.math.add(third[:,:,:,0]*255,-103.939)
    third[:,:,:,1]=tf.math.add(third[:,:,:,1]*255,-116.779)
    third[:,:,:,2]=tf.math.add(third[:,:,:,2]*255,-123.68)
    
    fourth[:,:,:,0]=tf.math.add(fourth[:,:,:,0]*255,-103.939)
    fourth[:,:,:,1]=tf.math.add(fourth[:,:,:,1]*255,-116.779)
    fourth[:,:,:,2]=tf.math.add(fourth[:,:,:,2]*255,-123.68)
    
    X[:,:,:,0]=tf.math.add(X[:,:,:,0]*255,-103.939)
    X[:,:,:,1]=tf.math.add(X[:,:,:,1]*255,-116.779)
    X[:,:,:,2]=tf.math.add(X[:,:,:,2]*255,-123.68)
    # encode to one-hot
    return x_test_o,X,first,second,third,fourth
CALLBACKS = []
MODEL_PATH = 'pgd/models/cifar10-n-deep'
model_name='/Cifar10_adv'
optimizer = keras.optimizers.Adam()
inputs = keras.Input(shape=(32, 32, 3))
upscale = keras.layers.Lambda(lambda x: tf.compat.v1.image.resize_image_with_pad(x,
                              160,
                              160,
                              method=tf.image.ResizeMethod.BILINEAR))(inputs)
model=keras.models.load_model(MODEL_PATH+model_name)


y_test = keras.utils.to_categorical(y_test, 10)
x_test_o,X,first,second,third,fourth= preprocess_data(x_test,y_test,model,GPU_ID)
print("\n\n\nDONE\n\n\n")
# upscale layer
eval_results_o=model.evaluate(x_test_o, y_test, batch_size=64, verbose=0)
eval_results_adv=model.evaluate(X, y_test, batch_size=64, verbose=0)
eval_results_1=model.evaluate(first, y_test, batch_size=64, verbose=0)
eval_results_2=model.evaluate(second, y_test, batch_size=64, verbose=0)
eval_results_3=model.evaluate(third, y_test, batch_size=64, verbose=0)
eval_results_4=model.evaluate(fourth, y_test, batch_size=64, verbose=0)
print("Natural Image Accuracy:\n",eval_results_o[1],"\n")
print("Adversarial Image Accuracy:\n",eval_results_adv[1],"\n")
print("First Autoenocoded Image Accuracy:\n",eval_results_1[1],"\n")
print("Second Autoenocoded Image Accuracy:\n",eval_results_2[1],"\n")
print("Third Autoenocoded Image Accuracy:\n",eval_results_3[1],"\n")
print("Fourth Autoenocoded Image Accuracy:\n",eval_results_4[1],"\n")