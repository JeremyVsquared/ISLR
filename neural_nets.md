# Neural Networks

Neural networks have been proven to be a universal approximator. That is, for any given input $x$ and any given output $y$, a neural network can be trained to approximate this function. This is known as a _universality theorem_. 

## Architectural Design

Every neural network has one _input layer_ and one _output layer_, and may have _hidden layers_ between them. The _input layer_ will have a number of neurons equal to the number of features in the training data, with some adding a node for a _bias term_. As for the _output layer_, if the neural network is a regressor it will have one node, and if it is a classifier it will have either a single node or one node per category.

As for the hidden layers, if the data is linearly separable then no hidden layers are necessary. If not, a single hidden layer is quite sufficient for vast majority of situations. The number of nodes within the single hidden layer is typically some number between the number of input nodes and output nodes.

## Strengths & Weaknesses

Neural networks perform particularly well at modeling rational differences and ratios, and quite poorly with rational polynomials.

# Convolutional Neural Networks

_Convolutional Neural Networks_ are deep neural networks with hidden, convolutional layers that specifically target characteristics of the features. As an example, a convolutional layer being used for computer vision may be looking for straight lines, angles, arcs, or light or dark areas. These layers then pass their output to the next layer as an input such that, for instance, identifying a series of co-located arcs would be identified in the next layer as a circle or angles as a square. The convolutional layers may continue to build these basic characteristics into basic shapes into combinations of shapes into more complicated combinations, and these are eventually used to identify likelihood of the various classes such as a bird or a person or a car. These likelihoods are then passed to a fully connected layer which chooses the class with the highest percentage.

## References

- [An Empirical Analysis fo Feature Engineering for Predictive Modeling](https://arxiv.org/pdf/1701.07852.pdf)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)