import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import os
import random
import pickle

DataDir = 'Frutas'
Categorias = ['Banana','Apple','Tomate']
IMG_SIZE = 50


def CreateTrainingData(Categorias,DataDir,pxSIZE):
	TrainingData = []
	for Categoria in Categorias:
		path = os.path.join(DataDir,'Training',Categoria)
		print('Carregando '+Categoria)
		class_num = Categorias.index(Categoria)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
				# plt.imshow(img_array,cmap='gray')
				# plt.show()
				newimg_array =cv2.resize(img_array,(pxSIZE,pxSIZE))
				#plt.imshow(newimg_array,cmap='gray')
				#plt.show()
				TrainingData.append([newimg_array,class_num])
			except Exception as e:
				print('erro carregando imagem '+os.path.join(path,img))
				#print(e)
	random.shuffle(TrainingData)
	return TrainingData

TData = CreateTrainingData(Categorias,DataDir,IMG_SIZE)

for sample in TData[:10]:
	print(Categorias[sample[1]])

x = []
y = []

for feature, label in TData:
	x.append(feature)
	y.append(label)

x = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,1)

pickle_out = open('x.pickle','wb')
pickle.dump(x,pickle_out)
pickle_out.close()

pickle_out = open('y.pickle','wb')
pickle.dump(y,pickle_out)
pickle_out.close()