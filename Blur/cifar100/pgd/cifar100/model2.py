import numpy as np
import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers

class Model:
    def __init__(self,training):
        epsilon=.0001
        #Input Layer
        inputlay=layers.Input(shape=[32,32,1])
        inputnorm=layers.BatchNormalization(epsilon=epsilon,center=True,scale=True, \
                                            beta_initializer='zeros', gamma_initializer='ones',\
                                                moving_mean_initializer='zeros',moving_variance_initializer='ones',\
                                                    renorm=True,trainable=training)(inputlay)
        relu1=layers.ReLU(negative_slope=.1)(inputnorm)
        conv1=layers.Conv2D(100,(3,3),strides=(1,1),padding='same')(relu1)
        
        
        
        #Layer One
        batchnorm2=layers.BatchNormalization(epsilon=epsilon,center=True,scale=True, \
                                            beta_initializer='zeros', gamma_initializer='ones',\
                                                moving_mean_initializer='zeros',moving_variance_initializer='ones',\
                                                    renorm=True,trainable=training)(conv1)
        relu2=layers.ReLU(negative_slope=.1)(batchnorm2)
        conv2=layers.Conv2D(128,(3,3),strides=(1,1),padding='same')(relu2)
        max_pool2=layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')(conv2)
        
        
        
        #Resnet Layers
        for i in range(2):
            #First Layer
            if(i==0):
                batchnorm3=layers.BatchNormalization(epsilon=epsilon,center=True,scale=True, \
                                                    beta_initializer='zeros', gamma_initializer='ones',\
                                                        moving_mean_initializer='zeros',moving_variance_initializer='ones',\
                                                            renorm=True,trainable=training)(max_pool2)
                relu3=layers.ReLU()(batchnorm3)
                conv3=layers.Conv2D(128,(3,3),strides=(1,1),padding='same')(relu3)
            else:
                merge_layer=layers.Add()([conv3,max_pool2])
                batchnorm3=layers.BatchNormalization(epsilon=epsilon,center=True,scale=True, \
                                                beta_initializer='zeros', gamma_initializer='ones',\
                                                    moving_mean_initializer='zeros',moving_variance_initializer='ones',\
                                                        renorm=True,trainable=training)(merge_layer)
                relu3=layers.ReLU()(batchnorm3)
                conv3=layers.Conv2D(128,(3,3),strides=(1,1),padding='same')(relu3)
            
            
            #Second Layer
            batchnorm4=layers.BatchNormalization(epsilon=epsilon,center=True,scale=True, \
                                                beta_initializer='zeros', gamma_initializer='ones',\
                                                    moving_mean_initializer='zeros',moving_variance_initializer='ones',\
                                                        renorm=True,trainable=training)(conv3)
            relu4=layers.ReLU()(batchnorm4)
            conv4=layers.Conv2D(128,(3,3),strides=(1,1),padding='same')(relu4)
            
            #Third Layer
            batchnorm5=layers.BatchNormalization(epsilon=epsilon,center=True,scale=True, \
                                                beta_initializer='zeros', gamma_initializer='ones',\
                                                    moving_mean_initializer='zeros',moving_variance_initializer='ones',\
                                                        renorm=True,trainable=training)((conv4))
            relu5=layers.ReLU()(batchnorm5)
            conv5=layers.Conv2D(128,(3,3),strides=(1,1),padding='same')(relu5)
            max_pool2=layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')(conv5)
            
            
        batchnorm6=layers.BatchNormalization(epsilon=epsilon,center=True,scale=True, \
                                                beta_initializer='zeros', gamma_initializer='ones',\
                                                    moving_mean_initializer='zeros',moving_variance_initializer='ones',\
                                                        renorm=True,trainable=training)(max_pool2)
        relu6=layers.ReLU()(batchnorm6)
        conv6=layers.Conv2D(32,(3,3),strides=(1,1))(relu6)
        max_pool6=layers.MaxPool2D(pool_size=(2,2),strides=(1,1))(conv6)
            
            
        flat=layers.Flatten()(max_pool6)
        
        dense1= layers.Dense(512,activation='relu', use_bias=True)(flat)
        dense4= layers.Dense(1024,activation='relu', use_bias=True)(dense1)
        finallay= layers.Dense(100, use_bias=True)(dense4)
        
        self.model=tf.keras.Model(inputs=inputlay,outputs=finallay,name="Model")
        self.model.compile(optimizer="adam",loss='mean_squared_error',metrics=['accuracy'])
        
        

        