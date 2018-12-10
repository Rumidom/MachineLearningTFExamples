import tensorflow as tf
import pickle
import cv2
import numpy as np
import os

Categorias = ['Banana','Apple','Tomate']
IMG_SIZE = 100

def prepara(filepath,pxSIZE):
	img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
	newimg_array =cv2.resize(img_array,(pxSIZE,pxSIZE))

	return np.array(newimg_array).reshape(-1,pxSIZE,pxSIZE,1)

model = tf.keras.models.load_model('CNN.model')

path = os.path.join('Test','Apple')
for img in os.listdir(path):
	prediction = model.predict([prepara(os.path.join(path,img),IMG_SIZE)])
	print('IMG: '+img)
	print('Banana: '+str(round(prediction[0][0])))
	print('Maçã: '+str(round(prediction[0][1])))
	print('Tomate: '+str(round(prediction[0][2])))
	print('-------')