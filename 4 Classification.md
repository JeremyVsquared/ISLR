# 4 Classification

Methods of classification often do so by predicting the probability of a given observation belonging to a particualr category, much like regression with buckets of value ranges. The 3 most popular classifiers are _logistic regression_, _linear discriminant analysis_, and _k-nearest neighbors_.

___

## Logistic Regression
Logistic regression measures the probability that $Y$ belongs to a particular category. I.e., if predicting the likelihood of default from the balance of an account, this would be written

$Pr(default=Yes | balance)$

Abbreviated $p(balance)$ and ranges from 0 to 1. The _decision boundary_ can be set anywhere within this range depending upon circumstances and the desired confidence level in the categorization.

### The Logistic Model
The relationship $p(X)=Pr(Y=1|X)$ and $X$ must necessarily be modeled by a function that outputs values between 0 and 1. There are a variety that can be used, but the _logistic function_ is used in the case of logistic regression.

$P(X)=\frac{e^{\beta_0+\beta_1x}}{1+e^{\beta_0+\beta_1x}}$

This function will output values that are certain to never exceeed the range of 0 to 1. This can be altered to

$\frac{p(x)}{1*p(x)}=e^{\beta_0+\beta_1x}$

The left side of the equation is called the _odds_ and can be between 0 and $\infty$. Taking the logarith of both sides gives us

$Log(\frac{p(x)}{1*p(x)})=\beta_0+\beta_1x$

The left-hand side is the _log-odds or logit_. In linear rergression, $\beta_1$ is the average change in $Y$ for each unit change in $X$. In logistic regression, each unit of change in $X$ corresponds to $\beta_1$ change in log odds, or multiplies the odds by $e^{\beta_1}$.

### Estimating the Regression Coefficients
Estimating the coefficients is best done with the method of _maximum likelihood_. We want to find $\beta_0$ and $\beta_1$ such that using the estimates in the model $p(X)$ yields an output pretty close to the observed value. This is formalized by maximizing the likelihood function

$\ell(\beta_0, \beta_1)=\prod_{i:y_i=1}p(x_i)\prod_{i':y'_i=0}(1-p(x_p))$

The least squares approach in linear regression is a special case of maximum likelihood.

### Multiple Logistic Regression
Predicting classification when presented with more than 2 classes is a bit different from 2-class, but can still be done using the maximum likelihood method.

There are circumstances where one might run multiple 2-class regressions on the same topic that can produce conflicting conclusions. This effect is called _confounding_.

>Ex, consider predicting multiple 2-class regressions to predict default rates based upon a number of factors: balance, student/non-student, etc. Then consider that separate classification regressions determine that students carry more debt than non students and those carrying more debt are more likely to default. These results lead to ambiguous, conflicting conclusions then would be produced if all factors were considered in a single classification.

Logistic regression with more than 2 response classes tend not to be used often in favor of methods such as discriminant analysis.

## Linear Discriminant Analysis
In logistic regression, we directly model $Pr(Y=k|X=x)$ for 2 response classes. In linear discriminant analysis, we model the distribution of the predictors $X$ separately in each of the response classes, and then use Bayes' to flip these around into estimates for $Pr(Y=k|X=x)$.

3 reasons why linear discriminant analysis can be useful:
1. When the classes are well-separated, the parameter estimates for the logistic regression model are surprisingly unstable. Linear discriminant analysis does not suffer from this problem.
2. If $n$ is small and the distribution of the predictors $X$ is approximately normal in each of the classes, the linear discriminant model is again more stable than the logistic regression model.
3. Linear discriminant analysis is a popular choice when there are more than 2 response classes.

### Using Bayes' Theorem for Classification
Let _$\pi_k$ represent the overall of prior probability_ that a randomly chosen observation comes from the $k^{th}$ class; this is the probability that a given observation is associated with the $k^{th}$ category of the response variable $Y$. Let $f_k(X)=Pr(X=x|Y=k)$ denote the _density function_ of $X$ for an observation that comes from the $k^{th}$ class. In other words, _$f_k(x)$ is relatively large if there is a high probability that an observation in the $k^{th}$ class has $X\approx x$, and $f_k(x)$ is small if it is very unlikely that an observation in the $k^{th}$ class $X\approx x$_. Then Bayes' Theorem states that

$Pr(Y=k|X=x)=\pi_kf_k(x)/\sum^k_{i=1}\pi_kf_k(x)$

_$p_k(x)$ is the posterior probability_ that an observation $X=X$ belongs to the $k^{th}$ class. Ie, it is the probability that a given observation belongs to the $k^{th}$ class, given the predictor value for that observation.

### Linear Discriminant Analysis for p=1
Assume that $p=1$ (there is only one predictor). The goal is to obtain an estimate for $f_k(x)$ that we can plug into in order to estimate $p_k(x)$. We will then classify an observation to the class for which $p_k(x)$ is greatest.

Suppose we assumed that $f_k(x)$ is normal or Gaussian. In a one-dimensional setting, the normal density takes the form

$f_k(x)=\frac{1}{\sqrt{2\pi\sigma_k}exp(-\frac{1}{2\sigma^2_k}(x-\mu_k)^2)}$

Where $\mu_k$ and $\sigma^2_k$ are the mean and variance parameters for the $k^{th}$ class. Assuming a shared variance term (ie, $\sigma^2_1=...=\sigma^2_k$) and some algebraic rearrangement, we arrive at assigning the greatest class from the following

$\delta_k(x)=x*\frac{\mu^2_k}{\sigma}-\frac{\mu^2_k}{2\sigma^2}+log(\pi_k)$

Ie, if $K=2$ and $\pi_1=\pi_2$, then the Bayes classifier assignment observation to class 1 if $2x(\mu_1-\mu_2)>\mu^2_1-\mu^2_2$, and to class 2 otherwise. In this case, the Bayes decision boundary corresponds to the point where

$x=\frac{\mu^2_1-\mu^2_2}{2(\mu_1-\mu_2)}=\frac{\mu_1+\mu_2}{2}$

The linear discrimant analysis (LDA) method approximates the Bayes classifier by plugging estimates for $\pi_k$, $\mu_k$, and $\sigma^2$ as

$\hat{\mu_k}=\frac{1}{n_k}\sum_{i:y_i=k}x_i$

$\hat{\sigma^2}=\frac{1}{n-K}\sum^k_{k=1}\sum_{i:y_i=k}(x_i-\hat{\mu_k})^2$

Where $n$ is the total number of training observations, and $n_k$ is the number of training observations in the $k^{th}$ class. The estimate $\mu_k$ is simply the average of all the training observations from the $k^{th}$ class, while $\hat{\sigma^2}$ can be seen as a weighted average of the sample variances for each of the $K$ classes

$\hat{\pi_k}=\frac{n_k}{n}$

The LDA classifier plugs the estimates given above into the prediction formula, and assigns an observation $X=X$ to the class for which 

$\hat{\sigma_k}(x)=x*\frac{\hat{\mu^2_k}}{\hat{\sigma}}-\frac{\hat{\mu^2_k}}{2\hat{\sigma^2}}+log(\hat{\pi_k})$

is largest. The word linear in the classifier's name stems from the fact that the discriminant functions $\hat{\delta_k}(x)$ in the above are linear functions of $x$ (as opposed to a more complex function of $x$).

### Linear Discriminant Analysis for p>1
In order to extend LDA to multiple predictors, we will assume that $X=(X_1, X_2, ..., X_p)$ is drawn from a multivariate Gaussian (or multivariate normal) distribution, with a class-speicific mean vector and a common covariance matrix.

For $p>1$ predictor LDA classifier assumes that the observations in the $k^{th}$ class are drawn from multivariate Gaussian distribution $N(\mu_k, \sum)$ where $\mu_k$ is a class-specific mean vector, and $\sum$ is a covariance matrix that is common to all $K$ classes.

_For $p>1$ LDA classifiers, there will actually be a decision boundary for each pair of classes_. Ie, if there are 3 classes A, B, and C, there will be 3 decision boundaries for $A|B$, $B|C$, and $A|C$. So $p>1$ classifiers effectively run multiple classifications in order to narrow down the classification.

Class specific performance is often evaluated by _sensitivity_ and _specificity_. Sensitivity is the percentage of actual class observations that are classified as such, whereas specificity is the percentage of the opposite class that are identified as such.

The ROC curve (Receiver Operating Characteristics) is a popular tool for graphical analysis of false positives and false negatives. The overall performance of a classifier, summarized over all possible thresholds, is given by the area under the ROC curve, known as the AUC. An ideal ROC curve will hug the top left corner, maximizing the AUC. A classifier with no better results than chance (50/50) will have an AUC of 0.5 as it is 45 degree diagonal line bisecting the area of the graph.

Varying the classifier threshold changes the true positive and false positive rate, which is known as the _sensisitivity_ and one minus the _specificity_ of the classifier. Where the threshold should properly be placed is subject and a matter of domain knowledge rather than absolutes.

### Quadratic Discriminant Analysis
Like LDA, QDA classifier results from assuming that the observations from each class are drawn from a Gaussian distribution, and plugging estimates for the parameters into Bayes' theorem in order to perform prediction. They differ in that QDA assumes that each class has its own covariance matrix. Formally, it assumes that an observation from $k^{th}$ class is of the form $X\approx N(\mu_k, \sum_k)$ where $\sum$ is a covariance matrix for the $k^{th}$ class.

One might choose between LDA and QDA due to the bias-variance trade off. LDA is much less flexible so it has lower variance but can suffer from high bias. QDA on the other hand is better when there is a large training set so that the variance of the classifier is not a major concern.