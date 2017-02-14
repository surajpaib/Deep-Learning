from tflearn.data_utils import  to_categorical, pad_sequences
from tflearn.datasets import imdb
import tflearn

train, test, _ = imdb.load_data(path='imdb.pkl', n_words=1000
0,
                                valid_portion=0.1)
trainX, trainY = train
testX, testY = test

trainX = pad_sequences(trainX, maxlen=100, value=0.)
trainY = to_categorical(trainY, 2)
testX = pad_sequences(testX, maxlen=100, value=0.)
testY = to_categorical(testY, 2)
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000,output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.0001)

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, show_metric=True, n_epoch=1000, batch_size=20, validation_set=(testX, testY))
