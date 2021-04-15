import numpy as np
import tensorflow.compat.v1 as tf;
import random
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers

class Model:
    def __init__(self,training):
        epsilon=.001
        inputlay=layers.Input(shape=[28,28,1])
        inputnorm=layers.BatchNormalization(axis=3,epsilon=epsilon,center=True,scale=True, \
                                            beta_initializer='zeros', gamma_initializer='ones',\
                                                moving_mean_initializer='zeros',moving_variance_initializer='ones',\
                                                    renorm=True,trainable=training)(inputlay)
        relu1=layers.ReLU(negative_slope=.1)(inputnorm)
        conv1=layers.Conv2D(100,(3,3),strides=1,padding='same')(relu1)
        
        
        
        
        batchnorm2=layers.BatchNormalization(axis=3,epsilon=epsilon,center=True,scale=True, \
                                            beta_initializer='zeros', gamma_initializer='ones',\
                                                moving_mean_initializer='zeros',moving_variance_initializer='ones',\
                                                    renorm=True,trainable=training)(conv1)
        relu2=layers.ReLU(negative_slope=.1)(batchnorm2)
        conv2=layers.Conv2D(100,(3,3),strides=1,padding='same')(relu2)
        max_pool2=layers.MaxPool2D(pool_size=(2,2),padding='same')(conv2)
        
        
        
        
        batchnorm3=layers.BatchNormalization(axis=3,epsilon=epsilon,center=True,scale=True, \
                                            beta_initializer='zeros', gamma_initializer='ones',\
                                                moving_mean_initializer='zeros',moving_variance_initializer='ones',\
                                                    renorm=True,trainable=training)(max_pool2)
        relu3=layers.ReLU()(batchnorm3)
        conv3=layers.Conv2D(64,(3,3),strides=1,padding='same')(relu3)
        max_pool3=layers.MaxPool2D(pool_size=(2,2),padding='same')(conv3)
        
        flat=layers.Flatten()(max_pool3)
        
        dense1= layers.Dense(1024,activation='relu', use_bias=True)(flat)
        dense2= layers.Dense(524,activation='relu', use_bias=True)(dense1)
        finallay= layers.Dense(10, use_bias=True)(dense2)
        
        self.model=tf.keras.Model(inputs=inputlay,outputs=finallay,name="Model")
        self.model.compile(optimizer="adam",loss='mean_squared_error')
        