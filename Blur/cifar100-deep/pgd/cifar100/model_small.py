import numpy as np
import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers

def resid_bloc(merge_layer,training):
        epsilon=.0001
        batchnorm1=layers.BatchNormalization(epsilon=epsilon,center=True,scale=True, \
                                                beta_initializer='zeros', gamma_initializer='ones',\
                                                    moving_mean_initializer='zeros',moving_variance_initializer='ones',\
                                                        renorm=True,trainable=training)(merge_layer)
        relu1=layers.ReLU()(batchnorm1)
        conv1=layers.Conv2D(64,(3,3),strides=(1,1),padding='same')(relu1)
                
            
            
            #Second Layer
        batchnorm2=layers.BatchNormalization(epsilon=epsilon,center=True,scale=True, \
                                                beta_initializer='zeros', gamma_initializer='ones',\
                                                    moving_mean_initializer='zeros',moving_variance_initializer='ones',\
                                                        renorm=True,trainable=training)(conv1)
        relu2=layers.ReLU()(batchnorm2)
        conv2=layers.Conv2D(64,(3,3),strides=(1,1),padding='same')(relu2)
        merge_layer=layers.Add()([conv2,merge_layer])
        return merge_layer

class Model:
    

    
    def __init__(self,training):
        epsilon=.0001
        #Input Layer
        inputlay=layers.Input(shape=[32,32,1])
        inputnorm=layers.BatchNormalization(epsilon=epsilon,center=True,scale=True, \
                                            beta_initializer='zeros', gamma_initializer='ones',\
                                                moving_mean_initializer='zeros',moving_variance_initializer='ones',\
                                                    renorm=True,trainable=training)(inputlay)
        convinit=layers.Conv2D(64,(7,7),strides=(2,2),padding='same')(inputnorm)
        reluinit=layers.ReLU()(convinit)
        merge_layer=layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')(reluinit)
        
        
        
        
        
        #Resnet Layers
        for i in range(2):
            merge_layer=resid_bloc(merge_layer,training)
        for i in range(2):
            merge_layer=resid_bloc(merge_layer,training)
            

        relu3=layers.ReLU()(merge_layer)
        pool_final=layers.GlobalAveragePooling2D()(relu3)
            
            
        flat=layers.Flatten()(pool_final)

        self.finallay= layers.Dense(100, use_bias=True)(flat)
        softmax=layers.Softmax()(self.finallay)
        
        self.model=tf.keras.Model(inputs=inputlay,outputs=softmax,name="Model")
    
        

        