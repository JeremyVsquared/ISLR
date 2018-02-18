- reasons to use embeddings:
  - one-hot encoded vectors are high-dimensional and sparse and embeddings reduces dimensionality and maintaining consistent dimensionality (in the context of variable length inputs)
  - vectors of each embedding get updated while training the neural network which allows visualizing relationships in multi-dimensional space

- creating an embedding matrix
  - decide how many __latent factors__ are used (how long we want the vector to be)
  - instead of ending up with huge one-hot encoded vectors, we can use an embedding matrix to keep the size of each vector much smaller
  - not every word gets replaced by a vector; instead, it gets replaced by index that is used to look up the vector in the embedding matrix

- recommender systems
  - 2 main types of recommender systems
    - __content-based filtering__: based on data about the item/product
    - __collaborative filtering__: find other users like a given user, see what they liked and assume the given user likes the same
  - in order to solve this problem we can create a huge matrix of the ratings of all users against all items/products, but in many cases this will create an extremely sparse matrix; this can be reduced by using an embedding rather than a sparse matrix

- word embedding
  - is a class of approaches for representing words and documents using a dense vector representation
  - words are represented by dense vectors where a vector represents the projection of the word into a continuous vector space
  - the position of a word in the learned vector space is referred to as its embedding
  - popular examples of methods of learning word embeddings from text: Word2Vec, GloVe

- keras embedding layer
  - keras offers an embedding layer 
  - requires that the input data be integer encoded, so that each word is represented by a unique integer; accomplished using tokenizer API
  - embedding layer is initialized with random weights, updated during training
  - can be used in a variety of ways:
    - alone to learn word embedding that can be saved and used in another model later
    - part of a deep learning model where it is learned along with the model itself
    - load a pre-trained word embedding model as a type of transfer learning
  - the embedding layer is defined as the first hidden layer of a network, must specify 3 arguments:
    - **input_dim**: size of the vocabulary in the text data
    - **output_dim**: size of the vector space in which words will be embedded; size of the output vectors from this layer for each word
    - **input_length**: length of input sequences, as you would define for any input layer of a Keras model
  - for example, a vocab of 200, vector space of 32 dimensions, input of 50 words each will be defined as ```e = Embedding(200, 32, input_length=50)```
  - output is a 2D vector with one embedding for each word in the input sequence of words
  - if you wish to connect a dense layer directly to an embedding layer, you must first flatten the 2D output matrix to a 1D vector using the flatten layer

- example of learning an embedding

  ​

Embeddings are dense vectors of real numbers which can reduce or guarantee constant dimensionality of an input space, which is a feature representation of latent factors with which certain properties can be represented by notions of distance. These can be used in text to model semantic relationships in which each word or phrase is represented by a vector but can be conceptually evaluated. For instance, a trained word embedding may provide a visualization which places “mood” near “happy” and “sad”, but far from “Brazil” and “Canada”. Embeddings can be applied to any form of data including user movie ratings, server or user activity logs. Additionally, embedding layers can be integrated into deep networks.

Embeddings are exceptionally useful (1) in cases of inconsistent dimensionality, (2) can reduce the dimensionality from other methods (such as bag of words) while maintaining context (ie, word order), and (3) embeddings are trained like a normal model and thus do not require hand tuning.