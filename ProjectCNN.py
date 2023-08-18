import os
import pandas as pd

import cv2
import cupy as cp
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from tensorflow.keras.callbacks import TensorBoard

datos, metadatos = tfds.load('emnist', as_supervised= True, with_info= True )
clases = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
          'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# Transformar imagenes de entrenamiento

datos_entrenamiento = []

for i, (imagen,etiqueta) in enumerate(datos['train'].take(100000)):
    imagen = np.transpose(imagen.numpy().reshape((28,28)))
    imagen = imagen.reshape(28,28,1)
    datos_entrenamiento.append([imagen,etiqueta])

datos_pruebas = datos['test'].take(50000)

# Visualizar imagenes de entrenamiento
plt.figure(figsize=(10,5))

for i, (imagen,etiqueta) in enumerate(datos['train'].take(12)):
    imagen = cp.transpose(imagen.numpy().reshape((28,28)))
    imagen = cv2.resize(imagen, (56,56), interpolation=0)
    plt.subplot(2,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f'{clases[np.array(etiqueta)]}')
    plt.imshow(imagen,cmap='gray')
    plt.savefig('data.png')

x = []  # imagenes de entrada
y = []  # etiquetas

for imagen, etiqueta in datos_entrenamiento:
    x.append(imagen)
    y.append(etiqueta)

# Convertirlo de lista a array
x = np.array(x)
y = np.array(y)

tamaño_lote = 50

# Modelo denso 2 capas

modeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape= (28,28,1)),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(62,activation='softmax')
])

modeloDenso.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

historialDenso = modeloDenso.fit(x, y, epochs=7, validation_split= 0.1, batch_size= tamaño_lote)

# Modelo con capas convoluciones y agrupacion

modeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(62,activation='softmax')
])

modeloCNN.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

historialCNN = modeloCNN.fit(x, y, epochs=7, validation_split= 0.1, batch_size= tamaño_lote)

# Modelo con capas convoluciones, agrupacion y  dropout

modeloCNNDO = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(62,activation='softmax')
])

modeloCNNDO.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

historialCNNDO = modeloCNNDO.fit(x, y, epochs=7, validation_split= 0.1, batch_size= tamaño_lote)

plt.xlabel('Época')
plt.ylabel('Magnitud de Pérdida')
plt.plot(historialDenso.history['val_loss'], label = 'Denso')
plt.plot(historialCNN.history['val_loss'], label = 'CNN')
plt.plot(historialCNNDO.history['val_loss'], label = 'CNN&DO')
plt.legend()
plt.savefig('loss(3).png')
plt.show()

plt.xlabel('Época')
plt.ylabel('Magnitud de Pérdida')
plt.plot(historialCNN.history['val_loss'], label = 'CNN')
plt.plot(historialCNNDO.history['val_loss'], label = 'CNN&DO')
plt.legend()
plt.savefig('loss(2).png')
plt.show()

plt.xlabel('Época')
plt.ylabel('Magnitud de Precisión')
plt.plot(historialDenso.history['val_accuracy'], label = 'Denso')
plt.plot(historialCNN.history['val_accuracy'], label = 'CNN')
plt.plot(historialCNNDO.history['val_accuracy'], label = 'CNN&DO')
plt.legend()
plt.savefig('gain(3).png')
plt.show()

plt.xlabel('Época')
plt.ylabel('Magnitud de Precisión')
plt.plot(historialCNN.history['val_accuracy'], label = 'CNN')
plt.plot(historialCNNDO.history['val_accuracy'], label = 'CNN&DO')
plt.legend()
plt.savefig('gain(2).png')
plt.show()

for imagen_prueba, etiqueta_prueba in datos_pruebas.take(10):

    #Transformar imagen
    imagen_prueba = cp.transpose(imagen_prueba.numpy().reshape((28,28)))
    imagen_prueba = imagen_prueba.reshape(1,28,28,1)

    predictDenso = modeloDenso.predict(imagen_prueba)
    predictCNN = modeloCNN.predict(imagen_prueba)
    predictCNNDO = modeloCNNDO.predict(imagen_prueba)

    print(f'Real: {clases[etiqueta_prueba]}')
    print(f'Prediccion Denso: {clases[np.argmax(predictDenso)]}')
    print(f'Prediccion CNN: {clases[np.argmax(predictCNN)]}')
    print(f'Prediccion CNNNO: {clases[np.argmax(predictCNNDO)]}')
