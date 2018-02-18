- what to do when overfitting:
  - add more data
  - data augmentation
  - regularization
  - reduce architecture complexity



- adding more data
  - Producing fake data with a GAN
  - **pseudo-label**
    - using unlabeled data to tell us something about the structure of the data by semi-supervised learning
    - Practically, this is just predicting labels for unlabeled data based upon what was learned from the labeled data and using the newly labeled data as though the labels were true/accurate
    - for unlabeled data, _psuedo-labels_, just picking up the class which has the maximum predicted probability, are used as if they were true labels; this is in effect equivalent to _entropy regularization_
    - the proposed network is trained in a supervised fashion with labeled and unlabeled data simultaneously; for unlabeled data, pseudo-labels, just picking up the class which has the maximum predicted probability every weights update, are used as if they were true labels
    - Pseudo-label method for deep neural networks
      - Pseudo-label is a method for training deep neural networks in a semi-supervised fashion
      - DAE can be stacked to initialize deep neural networks
      - we use DAE in a supervised pre-training phase; masking noise with a probability of 0.5 is used for corruption
      - Pseudo-label are target classes for unlabeled data as if they were true labels; we just pick up the class which has maximum predicted probabilty for each unlabeled sample
    - why would it work?
      - the _cluster assumption_ states that the decision boundary should lie in low-density regions to improve generalization performance
      - because neighbors of a data sample have similar activations with the sample by embedding-based penalty term, it's more likely that data samples in a high-density region have the same label 
- **data augmentation**
  - a form of regularization by way of deliberately introducing alterations to the data for the purposes of generating new data, or new views into the data, while retaining the important underlying structure of the data
  - in the context of computer vision, this is accomplished by altering images by such methods as shifting it left and right, flipping it symmetrically horizontally and vertically, rotating by random degrees, random crops, slightly altering colors or changing color images to greyscale; in this way one image sample can be turned into dozens or more

# References

- [Pseudo-label](http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf)