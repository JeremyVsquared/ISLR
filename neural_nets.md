# Neural Networks

Neural networks are groups of interconnected processing nodes which perform basic calculations which when working together create powerful estimator. These filters, called __neurons__, are arranged in one or more __hidden layers__ between the input and the output, the specifications of which is referred to as an architecture. In general, the more complicated this architecture, the more complicated the relationship the network is capable of modeling.

```python
from sklearn.neural_network import MLPClassifier

# generate some simple training data
X_train = [[0., 0.], [1., 1.]]
y_train = [0, 1]
X_test = [[0.1, 0.2]]
y_test = [0]

# define & train the network
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1)
nn.fit(X_train, y_train)

nn.predict(X_test)
```

The above code defines a very simple network with 5 neurons in only one layer, set by the __hidden_layer_sizes__ parameter to __MPLClassifier()__. This simple network could be visualized as 2 nodes for the input, both of which are connected to all 5 nodes of the single hidden layer, all of which will be connected to the single output node.

![Simple neural network](images/nn_01_simple_network.png)

Computationally, what happens here is all inputs are passed to all neurons of the hidden layer. When passed, the input values are multiplied by a coefficient unique to each path, called __weights__, which is automatically tuned during training and the product of these values is what the neuron receives. Each neuron then evaluates the value it receives by some activation function which determines it's output. This activation function could result in such simple output as 0 or 1 or a scaled function , such as a sigmoid function, which will output some value between 0 or 1. Think of this as the difference between a light switch and a light dimmer. All of this is intended to matchematically model the way our brains function by neurons firing or not firing. The neural network was historically defined with neurons which output only 0 or 1, much like our own neurons. However, as they have been used and further developed in modern machine learning applications it has become clear that they are far more effective when they permit a scale of values between 0 and 1.

### Deep Neural Networks

The preceding network is rather simple. When dealing with a more complicated relationship than our example data above, we can increase number of neurons in the hidden layer from 5 to 10, or even 100. And if this is insufficient to improve the accuracy of the model, we can add more hidden layers. For instance, let's assume that we wanted increase the number of neurons from 5 to 8 and add another hidden layer. This could be done by altering our __MLPClassifier()__ definition to the follwoing:

```python
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8,8), random_state=1)
```

And now our network looks like this:

![Simple deep neural network](images/nn_02_deep_network.png)

In a single hidden layer scenario, all inputs are connected to all neurons in the hidden layer, and all of the hidden layer neurons are connected to the output. Similarly, in the case of multiple hidden layers, all neurons of each layer are connected to all neurons in the subsequent hidden layer. And just as was the case between the inputs and the first hidden layer, each connection between the first and second hidden layers will have their own unique weight which is multiplied by the output of the neurons in the first hidden layer, and the neurons in the second layer will perform their activation function upon these products before sending their output to the output neuron. Neural networks with more than one hidden layer are referred to as __deep neural networks__.

### Bias Terms

Thus far, our deep network will suffice to model some complicated relationships, but there is a fundamental flaw. If you think of the network model in terms of an equation, the only factors present are variables and coefficients. So what happens when it is given an input of 0? In this scenario, an input of 0 is guaranteed to produce an output of 0. This is desirable in certain circumstances, but certainly not all. As such, there must be some constant value, like an intercept term in a linear equation, added to the network architecture in order to guarantee a value in these situations. This intercept term is known as a __bias term__.

Bias terms are not connected to any neurons in previous layers and hold a constant value of 1. Like all other neurons in their layer, they do connect to each neuron in the subsequent layer thus providing an input effectively being the values of the weights of each connection.

Adding bias terms, our deep network can now be visualized like this:

![Deep neural network with bias terms](images/nn_03_bias.png)

### Architectural Design

In review, every neural network has one _input layer_, one _output layer_, and one or more _hidden layers_ between them. The _input layer_ will have a number of neurons equal to the number of features in the training data, with some adding a neuron for an input _bias term_. As for the _output layer_, if the neural network is a regressor it will have one node, and if it is a classifier it will have either a single node or one node per category.

As for the hidden layers, if the data is linearly separable then no hidden layers are necessary. Otherwise a single hidden layer is quite sufficient for the vast majority of situations. One should evaluate a variety of architectures in order to find an optimal solution.

## Training

Neural networks operate by perform simple calculations in each layer, passing these outputs as inputs into future layers in order to make predictions. Networks that operate in this way are called __feedforward__ networks. Before making these predictions, the network must first be trained. Training the neural network requires an algorithm that can interpret the prediction error by what's called the __cost function__ and make subtle changes the weights of the neurons. A small change in weights of any single neuron can spiral out to large portions of the network, sometimes casuing significant and unwanted changes to the output. In order to accomplish this, an algorithm can be used that can estimate how these changes impact the cost function and pass this estimation to the preceding neurons through the network in order to adjust the weights and hopefully improving the accuracy of the output. Improved accuracy is defined by $y^hat$ better approximating $y_i$, which is evaluated by the cost function, also referred to as the __loss__ or __objective__ function. This algorithm is called __backpropogation__. Backpropogation calculates the gradient of the cost function which is used to estimate better weights. Neurons were traditionally defined to output binary values, 0 or 1, which is referred to as a __perceptron__. This makes the network fairly brittle and requires larger, more complicated networks to better model relationships. Changing these neurons to be more flexible to make the network a better model, so __sigmoid__ neurons are more commonly used which output a ranged value between 0 and 1.

In order to estimate a set of weights which minimize the cust, gradient descent can be used but is not a very performant option. When facing larger networks, it can prove to be quite slow and cumbersome. This process can be sped up substantially by using __stochastic gradient descent__ which operates upon randomly sampled batches of data and extrapolating the results upon the larger data set.

### Strengths & Weaknesses

Neural networks have been proven to be a universal approximator. That is, for any given input $x$ and any given output $y$, a neural network can be trained to approximate this function. This is known as a _universality theorem_. Neural networks perform particularly well at modeling rational differences and ratios, and quite poorly with rational polynomials.

