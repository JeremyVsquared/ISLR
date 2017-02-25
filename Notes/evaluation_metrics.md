
# Sensitivity & Specifity

_Sensitivity_ is the calculation of the correct prediction of a true positive.

$$sensitivity = \frac{true\ positives}{true\ positives + false\ negatives}$$

_Specifity_ is the calculation of the correct prediction of true negatives.

$$specificity = \frac{true\ negatives}{true\ negatives + false\ positives}$$

# Precision & Recall

_Precision_ is the ratio of predicted true positives to actual positives. Can be thought of as a quantitative measure of how willing a model is to be wrong.

$$precision = \frac{true\ positives}{true\ positives + false\ positives}$$

_Recall_ is the ratio of predicted true positives to all positives. Can be thought of as the quantitative measure of how many positives are captured with the model regardless of how many incorrectly predicted positives are included.

$$recall = \frac{true\ positives}{true\ positives + false\ negatives}$$

# F-measure

The _F-measure_ is a combination of precision and recall. The common version of F-measure evaluates precision and recall as equal values, not prioritizing either over the other.

$$F-measure = \frac{2 * precision * recall}{recall + precision}$$

# AUC/ROC Curve

The _ROC curve (Receiver Operating Characteristic)_ evaluates the tradeoff between evaluating true positives and avoiding false positives. The _AUC (area under the curve)_ is used to evaluate the performance of the fit, and specifically how close it is to a perfect evaluation. The AUC value ranges from 0.5 (no predictive value) to 1.0 (perfect evaluation).

# R2

$r^2$, also referred to as the _coefficient of determination_, is an evaluation of how closely observations fit a regression.

$$R^2 = 1 - \frac{\sum_i(y_i - \hat{y}_i)^2}{\sum_i(y_i - \bar{y})^2} = \frac{\sum_i(\hat{y}_i - \bar{y})^2}{\sum_i(y_i - \bar{y})^2}$$

# Residual Sum of Squares

$$RSS = \sum_i (y_i - f(x_i))^2$$

# Mean Squared Errors

$$MSE = \frac{1}{n} \sum_i (\hat{y}_i - y_i)^2$$

___

# Methods for evaluating model performance

# k-Fold Cross-Validation

This method involves segmenting the data where each of $k$ segments is used as a validation set and the errors for each are averaged. This method permits the use of 100% of the data for training while still performing cross-validation. Analysis has shown that using $k > 10$ segments is unnecessary.

# Bootstrap Sampling

_Bootstrap sampling_ is a method of artificially increasing a dataset without collecting more observations. Functionally, it is repeatedly and randomly sampling from the dataset without consideration for whether or not a given observation has already been sampled. For example, a given dataset of 1000 observations could be used to generate multiple datasets of 1000 observations, each of which will contain duplicate observations. Bootstrapping is particularly advantageous over k-Fold Cross-Validation when working with smaller datasets.