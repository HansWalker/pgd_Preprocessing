import os
import matplotlib.pyplot as plt
# import torchvision.transforms as tt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.datasets import cifar10
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
load=0
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
device_name = tf.test.gpu_device_name()
print(device_name)
with tf.device('/GPU:0'):
    def preprocess_data(X, Y):
        X_p = keras.applications.resnet.preprocess_input(X)
    
        # encode to one-hot
        Y_p = keras.utils.to_categorical(Y, 10)
        return X_p, Y_p
    CALLBACKS = []
    MODEL_PATH = 'pgd/models/cifar10-n-deep'
    model_name='/Cifar10_Model2'
    optimizer = keras.optimizers.Adam()
    
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    # upscale layer
    inputs = keras.Input(shape=(32, 32, 3))
    upscale = keras.layers.Lambda(lambda x: tf.compat.v1.image.resize_image_with_pad(x,
                                  160,
                                  160,
                                  method=tf.image.ResizeMethod.BILINEAR))(inputs)
    resnet_model = keras.applications.ResNet50(include_top=False,
                                            weights='imagenet',
                                            input_tensor=upscale,
                                            input_shape=(160,160,3),
                                            pooling='max')
    out = resnet_model.output
    out = Flatten()(out)
    out = BatchNormalization()(out)
    out = Dense(256, activation='relu')(out)
    out = Dropout(0.3)(out)
    out = BatchNormalization()(out)
    out = Dense(128, activation='relu')(out)
    out = Dropout(0.3)(out)
    out = BatchNormalization()(out)
    out = Dense(64, activation='relu')(out)
    out = Dropout(0.3)(out)
    out = Dense(10, activation='softmax')(out)
    
    CALLBACKS.append(keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH,
                                                  monitor='val_acc',
                                                  save_best_only=True))
    
    CALLBACKS.append(keras.callbacks.EarlyStopping(monitor='val_acc',
                                                verbose=1,
                                                patience=5))
    
    CALLBACKS.append(keras.callbacks.TensorBoard(log_dir='logs'))
    
    model = keras.models.Model(inputs=inputs, outputs=out)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x=x_train,
              y=y_train,
              batch_size=128,
              epochs=40,
              callbacks=CALLBACKS,
              validation_data=(x_test, y_test))
    model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    model.save(MODEL_PATH+model_name)