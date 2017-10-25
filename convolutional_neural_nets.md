# Convolutional Neural Networks

_Convolutional Neural Networks_ are deep neural networks with hidden, convolutional layers that specifically target characteristics of the features. This methodology has proven to be very useful in computer vision applications, as well as natural language processing. The way it does so is by effectively examining an observation piece by piece by way of a sliding window which only reveals small portions of the whole, then reducing the data observed to a lower dimensional representation. This can be thought of as reading a sentence in an image by only looking at one letter at a time. If the model had to examine the entire sentence every time in order to interpret it, it would become very complex and require tremendous amounts of data for training in order to account for individual character differences between one observation and another. On the other hand, if the model only examines one character at a time, it only needs to be able to identify letters, numbers, and punctuation accurately rather than requiring every permutation of these components for a given written language.

A convolutional layer being used for a computer vision problem may be looking for straight lines, angles, arcs, or light or dark areas. These layers then pass their output to the next layer as an input such that identifying a series of co-located arcs would be identified in the next layer as a circle or angles as a square. The convolutional layers may continue to build these basic characteristics into basic shapes and further into combinations of shapes and further into more complicated combinations, and these are eventually used to identify likelihood of the various classes such as a bird, person or car. These likelihoods are then passed to a fully connected layer which chooses the class with the highest percentage.

```python
from __future__ import division, print_function, absolute_import

import tflearn
import tflearn.datasets.mnist as mnist
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# data load & prep
X_train, y_train, X_test, y_test = mnist.load_data(one_hot=True)
X_train = X_train.reshape([-1, 28, 28, 1])
X_test = X_test.reshape([-1, 28, 28, 1])

# build the CNN 
network = input_data(shape=[None, 28, 28, 1], name='input')
# 1st convolution layer
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
# 2nd convolution layer
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
# feed pooled, convoluted features into fully connected layers
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,loss='categorical_crossentropy', name='target')

# training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X_train}, {'target': y_train}, n_epoch=1,validation_set=({'input': X_test}, {'target': y_test}), snapshot_step=100, show_metric=True, run_id='convnet_mnist')
```

## Components

The convolutional neural network is a very powerful technique that is constituted of more than just a special layer in a neural network, but rather an architecture in and of itself. The CNN has four vital components: convolution, nonlinearity, pooling, and finally classification.

### Convolution

The sliding window concept of convolution was already discussed, but not in great detail. This sliding window, more commonly referred to as a __filter__, is looking for particular patterns within a portion of the input and collapsing this portion into a lower dimensional representation called a __convolved feature__, __activation map__ or __feature map__. As an oversimplified example, assume a filter that is looking for straight, horizontal lines. This filter will slide over the window left to right, top to bottom, and it's output will be a statement of whether or not it found a straight, horizontal line within the portion of the input it was examinging at that particular step. The distance of each step, such as the number of pixels in an image classification context, is called the __stride__ and is typically set to 1 in practice. The way this operation is performed is by element-wise multiplication on the filterd input, the outputs of which are added together to compute the final value. This value is then added to the output matrix spacially relative to the filter's position within the input observation. In this way, the dimensionality of the input is reduced while preserving spatial or sequential relationships of the features identified.

In thinking through the above mathematical operation, it should become obvious that features could be easily missed due to their position within the input when they are close to the edges. As such, it has become common pratice to use __zero-padding__ when this effect is seen. Zero-padding is the simple technique of surrounding the input with 0's, effectively creating a border such that features at the edges may be better examined. Operations with this zero-padding are called __wide convolutions__, while those without are called __narrow convolutions__.

This feature extraction from the input is somewhat limited if restrained to filtering for only one feature of the input. It is possible, and far more practical, to use multiple filters simultaneously so that the CNN as a whole can seek out numerous features. The number of filters used at any given time is referred to as the __depth__ of the convolution. The depth along with the window size, stride, and zero-padding defines the CNN from the user's perspective and all other values are discovered by the training operation. It is important to note that varying these values will produce different outputs, possibly identifying or missing features sought by the user.

### Non linearity

Most real world data is nonlinear in nature, but a careful reading of the dimension reduction operation of the convolution layer should reveal that this is a linear operation. In order to counter this, nonlinearity is deliberately re-introduced into the data by way of feeding the output of the convolution layer into a rectified linear unit, more commonly referred to as __ReLU__. Alternative nonlinear functions such as __tanh__ or __sigmoid__ have also been used, but ReLU has been found in practice to be the most effective.

### Pooling

Now that the data has been convolved and linearity corrected by ReLU, we further distill the data by __pooling__, or subsampling. This process further reduces the dimensionality of the data, thus making the input representations more manageable, while surfacing the most important features found by convolution. This has the effect of generating a representation of the input that is scale invariant. This is done by defining a spatial subset of feature map matrix and performing some operation upon those values to arrive at a single value representation. The most pooling method is __max pooling__ which simply outputs the greatest value within the ReLU feature map. Averaging and summing have also been used, but max pooling has been found to be more effective in practice. Pooling has the benefits of making the network resistant to noisy data and overfitting by reducing the computational load and parameters.

In the previous sample code, there was a call to local response normalization after the pooling function. This action, as the name would suggest, normalizes the pooled feature map so that areas of interest are more consistenly highlighted and easier to identify. Conceptually this can be thought of as a sort of tunnel vision, dampening surrounding activations and amplifying the strongest signals but also enforcing consistent values for the sake of processing efficiency.

### Fully connected layer

The pooling output is the data that the network will actually be modeling. The __fully connected layer__ is the final step before the expected output is returned and is just a neural network attached to the end of this chain of operations. This is hinted at by the phrasing "fully connected", which is intended to imply that every neuron within a given layer is attached to every neuron of the subsequent layer. 

In an image classification context that is classifying handwritten numbers, assume that the convolution layer identifies four contiguous arcs within the image, the pooling layer surfaces these values of possible background noise or weaker signals around these arcs, and these high level features of four contiguous arcs are fed into the fully connected layer which has 10 outputs comprising the probabilities of each class, summing to 1. The greatest probability value is used to classify this image as a "0".

## Training

Training a CNN takes place in 5 steps.

1. First, all filters and weights are intiialized with random values.
2. Next, the network progresses the first input through the network layers, convolution to ReLU to pooling to the fully connected layer, outputting the probabilities for each class.
3. Now that it has an output $\hat{y_i}$, the error is calculated against the true value $y_i$.
4. Backpropogation is used to compute the error gradient and the weights are adjusted accordingly.
5. Steps 2 through 4 are repeated upon the remainder of the training set.

## NLP

While CNN's have proven to be wildly effective in the field of computer vision, it is also useful in other contexts such as natural language processing. Due to the nature of the convolution and pooling operations, input sequence (word order) can be lost and as such using a CNN for entity extraction or part of speech tagging would not be advised, but it has been proven to be quite useful in such tasks as sentiment analysis and topic classification. While this may at first seem like a weakness, these aspects of the CNN also provide the benefits of guaranteeing dimensionality of possibly variable length inputs while also idealing surfacing pertinent information within the input.

Rather than a matrix of pixel values as one would see in computer vision, the input would be a matrix representation of the text by GloVe, one hot vectors, word2vec, etc. It is typical for the filter to cover 2 to 5 words at a time with a stride of 1. Greater strides can of course be used and will begin to approximat the behavior one would expect from an RNN.  A benefit of using CNN's for NLP is that the convolution filters are able to automatically develop sound representations without the necessity of maintaining the entire vocabulary. This reduces the computational load and enables the network to perform well upon n-grams larger than would ideal with other methods.

# More architectures

In general, the more convolution steps we have, the more complicated relationships our network will be able to learn to recognize

Notable architectures:
    - LeNet
    - AlexNet: deeper and wider version of LeNet; significant breakthrough in ImageNet visual recognition competition in 2012
    - ZF Net: improvement on AlexNet architecture; next winner of ImageNet competition in 2013
    - GoogLeNet: 2014 ImageNet winner; main contribution was the development of an __inception module__ that dramatically reduced the number of parameters in the network
    - VGGNet: runner up ImageNet in 2014; main contribution was in showing that the depth of the network (number of layers) is a critical component for good performance
    - ResNets: 2015 ImageNet winner
    - DenseNet: is a __densely connected convolutional network__ where each layer directly connected to every other layer in a feed-forward fashion; has been shown to obtain significant improvements over previous state of the art architectures