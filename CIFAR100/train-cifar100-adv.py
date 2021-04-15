import os
import matplotlib.pyplot as plt
# import torchvision.transforms as tt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.datasets import cifar100
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import math
load=0
device_name = tf.test.gpu_device_name()
print(device_name)
retrain=0
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
def preprocess_data(X, Y, model,epsilon):
    X_p = np.array(keras.applications.resnet.preprocess_input(X))
    epsilon*=255
    pert=epsilon/5
    loss = keras.losses.CategoricalCrossentropy()
    # encode to one-hot
    Y_p = keras.utils.to_categorical(Y, 100)
    if(epsilon!=0):
        for i in range(int(len(X_p)/100)):
            Z=tf.cast(X_p[i*100:(100*i+100)], tf.float32)
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
                X_p[100*i+k,:,:,0]=Z[k,:,:,0]
                X_p[100*i+k,:,:,1]=Z[k,:,:,1]
                X_p[100*i+k,:,:,2]=Z[k,:,:,2]
    return X_p, Y_p
CALLBACKS = []
MODEL_PATH = 'pgd/models'
model_name='/cifar100_adv'
if(retrain==0):
    optimizer = keras.optimizers.Adam()
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
    out = Dense(100, activation='softmax')(out)
    
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
if(retrain==1):
    model=keras.models.load_model(MODEL_PATH+model_name)
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

adv_pert=0.005

(x_train_o, y_train_o), (x_test_o, y_test_o) = cifar100.load_data()
for i in range(50000):
    x_train, y_train = preprocess_data(x_train_o, y_train_o,model,adv_pert)
    model.fit(x=x_train,
              y=y_train,
              batch_size=128,
              epochs=20,
              verbose=0)
    x_test, y_test = preprocess_data(x_test_o, y_test_o,model,adv_pert)
    acc=model.evaluate(x_test, y_test, batch_size=128, verbose=0)
    print(i)
    print("Model Accuracy: ",acc[1])
    print("Model pertubation: ",adv_pert)
    model.save(MODEL_PATH+model_name)
            