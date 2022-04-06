

from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.python.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D,TimeDistributed,Input
from tensorflow.keras.layers import Conv2DTranspose
import numpy as np
import pickle
import numpy
import os

#  Checking whether TPU is requested 
assert 'COLAB_TPU_ADDR' in os.environ,'Request for TPU in notebook setting'
if 'COLAB_TPU_ADDR' in os.environ:
    TF_MASTER='grpc://{}'.format(os.environ['COLAB_TPU_ADDR'])
else:
    TF_MASTER=''

tpu_address=TF_MASTER

resolver=tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)
tf.config.experimental_connect_to_cluster(resolver)
# Intialize TPU
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

"""Comparing training data with the ground """
'https://docs.python.org/3/library/pickle.html'
# Context Manager loading train data and test data
with open("t1.p","rb") as f:
    X_train1=pickle.load(f)
    X_gd1 = pickle.load( open( "g1.p", "rb" ) )
    X_train2 = pickle.load( open( "t2.p", "rb" ) )
    X_gd2 = pickle.load( open( "g2.p", "rb" ) )
    X_train=np.concatenate((X_train1,X_train2),axis=0)
    X_gd=np.concatenate((X_gd1,X_gd2),axis=0)
'n=number of images taken for evaluation'
X_gd=X_gd.reshape(n,1,192,640,1)
X_train=X_train.reshape(n,1,192,640,3)
'This architecture is based on the paper "DepthNet: A Recurrent Neural Network Architecture for Monocular Depth Prediction"'
from keras.layers import Add
with strategy.scope():
    inp = Input(shape=(None,192,640,3))
    x1=ConvLSTM2D(filters=32, kernel_size=(7, 7), 
                      padding='same', return_sequences=True)(inp)
    x=ConvLSTM2D(filters=64, kernel_size=(5, 5), 
                      padding='same', return_sequences=True)(x1)
 
    x=ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=(2, 2),
                      padding='same', return_sequences=True)(x)
 
    x2=(ConvLSTM2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', return_sequences=True))(x)
 
 
    x=(ConvLSTM2D(filters=128, kernel_size=(3, 3), strides=(2, 2),
                      padding='same', return_sequences=True))(x2)
 
    x3=(ConvLSTM2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', return_sequences=True))(x)
 
    x=(ConvLSTM2D(filters=256, kernel_size=(3, 3), strides=(2, 2),
                      padding='same', return_sequences=True))(x3)
 
    x4=(ConvLSTM2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', return_sequences=True))(x)
 
    y=TimeDistributed(Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=2, activation='relu', padding='same') )(x4)
    y=TimeDistributed(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))(y)
    y=(BatchNormalization())(y)

    y=TimeDistributed(Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=2, activation='relu', padding='same') )(x3)
    y=TimeDistributed(Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu'))(y)
    y=(BatchNormalization())(y)
    
    y=TimeDistributed(Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=1, activation='relu', padding='same') )(y)
    y= Add()([x2, y]) 

    y=TimeDistributed(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))(y)    
    y=(BatchNormalization())(y)
 
    y=TimeDistributed(Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='relu', padding='same') )(y)
    y=TimeDistributed(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))(y)
    y=(BatchNormalization())(y)
 
    y=TimeDistributed(Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=1, activation='relu', padding='same') )(y)

    y=TimeDistributed(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))(y)
    y=(BatchNormalization())(y)
 
    y=Conv2D(filters=1,kernel_size=(1,1),padding='same',activation='sigmoid')(y)
    y= Add()([x1, y]) 

    encoder = Model(inp,y,name='encoder')
    epochs = 25
    batch_size = 24
    encoder.summary()
    encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-05,epsilon=1e-07), loss='RMSE')
 
    history = encoder.fit(X_train, X_gd, batch_size=batch_size,
                          epochs=epochs, validation_split=0.05)

'https://docs.python.org/3/library/pickle.html'
X_test = pickle.load( open( "t27.p", "rb" ) )
predictions = np.zeros(shape=(160, *X_test[0].shape))
for i in range(0,30):
  frame=np.reshape(X_test[i],(1,1,192,640,3))
  val=encoder.predict(frame)
  val=val.reshape(192,640,1)
  val=val
  predictions[i]=val    

frame=np.reshape(X_train[4],(1,1,192,640,3))
val=encoder.predict(frame)
val=val.reshape(192,640)
'https://matplotlib.org/stable/tutorials/introductory/images.html'

import matplotlib.pyplot as plt
f, axes = plt.subplots(2,figsize=(20,10))
predictions=predictions*255
axes[0].imshow(predictions[8]/255,cmap='plasma')
axes[1].imshow(X_test[8],cmap='plasma')
