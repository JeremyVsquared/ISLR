# Recurrent Neural Networks

Recurrent neural networks, or RNN, operate in a very similar manner to neural networks with one important distinction. A recurrent neural network provides for communication between neurons of the same layer. In this way, a recurrent neural network retains information from the previous step of the training iteration. This adaptation makes the RNN especially useful when training on data that is dependent upon consecutive data such as text for natural language processing or time series wherein the proper interpretation of a given data point is often dependent upon past data points.

A weakness of the RNN is the rapid degradation of the signals passed into future data points. Due to the nature of the management of this data being passed forward, information inferred from a given step might only make it another step or two into the future steps. This problem is known as the _vanishing gradient_.

## LSTM-RNN

In order to address the problem of the _vanishing gradient_, an adaptation of the RNN was developed that would retain information longer and thus carrying information much further into future steps than was otherwise possible. This adaptation is the long short-term memory neural network and is the most popular variant of RNN. A LSTM neural network architecture actually has the ability to carry this information forward for arbitrary steps, evaluating the relevance of this information on each step to determine whether or not it should be retained further.

## Image Classification Application

Here we will use an LSTM to classify handwritten numbers as images to their respective digits, classified as 0 through 9. Here we are limiting the epochs to 1 as the __n_epoch__ parameter to __model.fit()__ in order to quickly see results. Increase the number epochs in order to improve accuracy.

'''{python}
from __future__ import division, print_function, absolute_import

import numpy as np
import tflearn

# import the dataset of images of handwritten numbers
import tflearn.datasets.mnist as mnist
X_train, y_train, X_test, y_test = mnist.load_data(one_hot=True)
# reshape input data
X_train = np.reshape(X_train, (-1, 28, 28))
X_test = np.reshape(X_test, (-1, 28, 28))

# build the LSTM RNN
net = tflearn.input_data(shape=[None, 28, 28])
net = tflearn.lstm(net, 128, return_seq=True)
net = tflearn.lstm(net, 128)
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net, optimizer='adam',
                         loss='categorical_crossentropy', name="output1")
model = tflearn.DNN(net, tensorboard_verbose=2)
# train the network
model.fit(X_train, y_train, n_epoch=1, validation_set=0.1, show_metric=True, snapshot_step=100)
'''

## Text Classification Application

Here we will use an LSTM to perform sentiment analysis on movie reviews from IMDB as positive or negative based upon the text content of the review. Here we are limiting the epochs to 1 as the __n_epoch__ parameter to __model.fit()__ in order to quickly see results. Increase the number epochs in order to improve accuracy.

'''{python}
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

# download the data set
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                valid_portion=0.1)
X_train, y_train = train
X_test, y_test = test

# prepare the data
X_train = pad_sequences(X_train, maxlen=100, value=0.)
X_test = pad_sequences(X_test, maxlen=100, value=0.)

y_train = to_categorical(y_train, nb_classes=2)
y_test = to_categorical(y_test, nb_classes=2)

# build the LSTM RNN
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=0)
# train the network
model.fit(X_train, y_train, n_epoch=1, validation_set=(X_test, y_test), show_metric=True, batch_size=32)
'''

# References
- [A Noobs Guide to Implementing RNN-LSTM Using Tensorflow](http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/)
- [Recurrent Neural Networks Tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)