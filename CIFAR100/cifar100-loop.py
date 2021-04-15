from tensorflow.keras.datasets import cifar100;
import tensorflow.keras as keras;
import tensorflow as tf;
tf.compat.v1.enable_eager_execution()
from PIL import Image
import numpy as np;
from autoencode_data import get_processed_data
import math
img_size1 = [32,32,3]
img_size2 = [32,32,1]
img_size1 = [32,32,3]
img_size2 = [32,32,1]
def preprocess_data(X,Y,model,epsilon,pert):
    epsilon=epsilon*255
    pert=pert*255
    X=np.array(keras.applications.resnet.preprocess_input(X))
    loss = keras.losses.CategoricalCrossentropy()
    for i in range(int(len(X)/100)):
        Z=tf.cast(X[i*100:(100*i+100)], tf.float32)
        Y_next=Y[i*100:(100*i+100)]
        if (epsilon!=0):
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
    X_p=get_processed_data(X_p, img_size2)
    X_p[:,:,:,0]=tf.math.add(X_p[:,:,:,0]*255,-103.939)
    X_p[:,:,:,1]=tf.math.add(X_p[:,:,:,1]*255,-116.779)
    X_p[:,:,:,2]=tf.math.add(X_p[:,:,:,2]*255,-123.68)
    
    X[:,:,:,0]=tf.math.add(X[:,:,:,0]*255,-103.939)
    X[:,:,:,1]=tf.math.add(X[:,:,:,1]*255,-116.779)
    X[:,:,:,2]=tf.math.add(X[:,:,:,2]*255,-123.68)
    # encode to one-hot
    return X_p,X
max_ep=.2
MODEL_PATH = 'pgd/models'
model_name='/cifar100'
number_of_runs=20
optimizer = keras.optimizers.Adam()
inputs = keras.Input(shape=(32, 32, 3))
upscale = keras.layers.Lambda(lambda x: tf.compat.v1.image.resize_image_with_pad(x,
                            160,
                            160,
                            method=tf.image.ResizeMethod.BILINEAR))(inputs)
model=keras.models.load_model(MODEL_PATH+model_name)
loop_list_adv=np.zeros([2,number_of_runs])
loop_list_auto=np.zeros([2,number_of_runs])
for i in range(3):
    for j in range(number_of_runs):
        (x_train, y_train), (x_test, y_test) = cifar100.load_data();
        epsilon=(j)*max_ep/(number_of_runs-1)
        pert=epsilon/10
        y_test = keras.utils.to_categorical(y_test, 100)
        x_test,X = preprocess_data(x_test,y_test,model,epsilon,pert)
        eval_results=model.evaluate(X, y_test, batch_size=64, verbose=0)
        eval_results2=model.evaluate(x_test, y_test, batch_size=64, verbose=0)
        loop_list_adv[0,j]=epsilon
        loop_list_adv[1,j]=eval_results[1]
        loop_list_auto[0,j]=epsilon
        loop_list_auto[1,j]=eval_results2[1]
    np.save('loop_data/cifar100_loop_list'+str(i)+".npy",loop_list_adv)
    np.save('loop_data/cifar100_loop_list_corr'+str(i)+".npy",loop_list_auto)