import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import pickle
import time

DataDir = 'Frutas'
Categorias = ['Banana','Apple','Tomate']

x = pickle.load(open('xtrain.pickle','rb'))
x = x/255.0

y = pickle.load(open('ytrain.pickle','rb'))

Name = 'Frutas CNN - {}'.format(int(time.time()))
TensorBoard = tf.keras.callbacks.TensorBoard(log_dir = 'logs/{}'.format(Name))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64,(3,3),input_shape = x.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(64,(3,3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Dense(3))
model.add(tf.keras.layers.Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x,y,batch_size=20,validation_split=0.10,epochs=18,callbacks=[TensorBoard])

ytest = pickle.load(open('ytest.pickle','rb'))
xtest = pickle.load(open('xtest.pickle','rb'))
val_loss,val_acc = model.evaluate(xtest,ytest)
print('----------------')
print('test val_loss = '+str(val_loss))
print('test val_acc = '+str(val_acc))
model.save('CNN.model')