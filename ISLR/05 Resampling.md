# 5 Resampling

Resampling methods involve repeatedly drawing samples from a training set and refitting a model of interest in order to obtain additional information about the fitted model. The two most commonly used resampling methods are _cross-validation_ and _bootstrap_. The process for evaluating a models performance is known as _model assessment_, whereas the process of selecting the proper level of flexibility for a model is known as _model selection_.

___

## Cross-validation
The difference between the test error and training error in evaluating predictions with previously unseen observations whereas the latter is the rate of error when training against the initial data. It is important to remember that the training error and test error can dramatically differ, and a good training error does not necessarily indicate a good test error.

The most common method of evaluating the test error is to withold a subset of the training data to be used for testing rather than training.

### The Validation Set Approach
This involves dividing the data randomly into the training set and a validation set. This approach is conceptually simple but has 2 potential drawbacks:
1. The test error can suffer from extremely high variability depending upon how the data set is divided.
2. Dividing the data set means fewer observations to train on which means a worse fit then could otherwise be found with more.

### Leave-One-Out Cross-Validation
Similar to the validation set approach but attempts to address the potential drawbacks. In LOOCV, a single observation is used as the test set while the $n-1$ set is used for training. This accounts for the problems of reducing the training set but using a single observation as a test set suffers from high variability. As such, the process is repeated, moving through the data set until every observation has been used as a test set. Then the error is averaged throughout.

LOOCV solves a few problems but can be expensive as it requires the model to be fit $n$ times. There is a shortcut in the following

$CV_n=\frac{1}{n}\sum^n_{i=1}(\frac{y_i-\hat{y_i}}{1-h_i})^2$

where $h$ is the leverage as defined by

$h_i=\frac{1}{n}+\frac{(x_i-x)^2}{\sum^n_{i'=1}(x_{i'}-\bar{x})^2}$

The leverage lies between $1/n$ and $1$ and reflects the amount that an observation influences its own fit.

### K-Fold Cross-Validation
Similar to LOOCV but the data set is divided into $k$ groups, or _folds_, of approximately equal size. Then the data is fit $k$ times on $k-1$ data, stepping through using the fold as a validation set each time and averaging the error. Effectively, LOOCV is k-fold cross-validation where $k=n$. The advantage of increasing $k$ is to save computational resources and reduce the number of necessary iterations.

An additional advantage is decreasing variability when compared to other methods.

### Bias-variance Trade-Off for k-Fold Cross-Validation
K-fold cross-validation often gives more accurate estimates of the test error rate than LOOCV due to the bias-variance trade-off. There is a bias-variance trade-off with the choice of k in k-fold cross-validation. Typically, given theese considerations, one performs k-fold cross-validation using $k=5$ or $k=10$, as these values have been shown empirically to yield test error rate estimates that suffer neither from excessively high bias nor from very high variance.

### Cross-Validation on Classification Problems
Cross-validation can be used in classification but in this context, rather than using MSE to quantify test error, we instead use the number of misclassified observations.

## Bootstrap
The bootstrap is a widely applicable and extremely powerful statistical tool that can be used to quantify the uncertainty associated with a given estimator or statistical learning method. This method generally refers to testing resampling means of _replacement_. Replacement refers to the resampling of data wherein observations may appear multiple times, or not at all.

Ex, Consider measuring fish in a static pool of water by mean sof catch and release. In doing so, there is no guarantee that the same fish will not be measured twice or that every fish will be caught and measured.