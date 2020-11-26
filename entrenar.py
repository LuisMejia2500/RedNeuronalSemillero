import os
import sys
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense,Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()

data_entrenamiento='./data/entrenamiento'
data_validacion='.data/validacion'

epocas=30

altura, longitud = 100, 100
batch_size=5
pasos=1000
pasos_validacion=20
filtrosConv1=32
filtrosConv2=64
tamaño_filtro1=(3,3)
tamaño_filtro2=(2,2)
tamaño_pool=(2,2)
clases=2
lr=0.00005

entrenamiento_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

data_validacion=ImageDataGenerator(
    rescale=1./255
)

imagen_entrenamiento=entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

imagen_validacion=validacion_datagen.flow_from_directory(
    data_validacion,
    target_size(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

cnn=Sequential()

cnn.add(Convolution2D(filtrosConv1, tamaño_filtro1, padding='save', input_shape=(altura, longitud) , activation='relu' ))
cnn.add(MaxPooling2D(pool_size=tamaño_pool))

cnn.add(Convolution2D(filtrosConv2, tamaño_filtro2, padding='save', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamaño_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.6))
cnn.add(Dense(clases, activation='softmax'))