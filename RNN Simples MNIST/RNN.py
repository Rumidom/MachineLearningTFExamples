import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense 
Dropout = tf.keras.layers.Dropout
LSTM = tf.keras.layers.LSTM

mnist =tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_test = x_test/255
x_train = x_train/255
#print(x_test.shape)
#print(x_train[0])

model = Sequential()
model.add(LSTM(128,input_shape=(x_train.shape[1:]),activation='relu',return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32,activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(10,activation = 'softmax'))

opt = tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5)

model.compile(loss = 'sparse_categorical_crossentropy',
	optimizer = opt,
	metrics =['accuracy']
	)

model.fit(x_train,y_train,epochs=3,validation_data=(x_test,y_test))
model.save('RNN_MNIST.model')