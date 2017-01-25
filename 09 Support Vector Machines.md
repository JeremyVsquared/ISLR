# 9 Support Vector Machines

Support vector machine is a generalization of a simple and intuitive classifer called the _maximal margin classifier_. The _support vector classifier_ is an extension of the maximal margin classifier that can be applied in a broader range of cases. The _support vector machine_ is a further extension of the support vector classifier in order to accomodate non-linear class boundaries.

___

## Maximal Margin Classifier

### What is a Hyperplane?

In a $p$-dimensional space, a _hyperplane_ is a flat affine subspace of dimension $p-1$. We can think of the hyperplane as dividing $p$-dimensional space into two halves.

### Classification Using a Separate Hyperplane

If a separating hyperplane exists, we can use it to construct a very natural classifier: a test observation is assigned a class depending on which side of the hyperplane it is located. A classifier that is based on a separating hyperplane leads to a linear decision boundary.

### The Maximal Margin Classifier

A natural choice for selecting one of the infinite options of hyperplane is the _maximal margin hyperplane_ (also known as the _optimal separating hyperplane_), which is the separating hyperplane that is farthest from the training observations. Although the maximal margin classifier is often successful, it can also lead to overfitting when $p$ is large.

Examining this when plotted, we see that three training observations are equidistant from the maximal margin hyperplane and lie along the dashed lines indicating the width of the margin. These three observations are known as _support vectors_, since they are vectors in $p$-dimensional space and they "support" the maximal margin hyperplane in the sense that if these points were moved slightly then the maximal margin hyperplane would move as well. The maximal classifier depends directly on the support vectors, but not on the other observations.

### Construction of the Maximal Margin Classifier

The maximal margin hyperplane is the solution to the optimization problem

1. $maximize_{\beta_0, \beta_1, ..., \beta_p}M$
2. Subject to $\sum^p_{j=1}\beta^2_j = 1$,
3. $y_i(\beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + ... + \beta_px_{ip}) \geq M \forall i = 1, ..., n$

(3) guarantees that each observation will be on the correct side of the hyperplane, provided that $M$ is positive. Note that (2) is not so much a constraint on the hyperplane, but rather giving definition to it. (2) and (3) working together ensure that each observation is on the correct side of the hyperplan and at least a distance $M$ from the hyperplane. Hence, $M$ represents the margin of our hyperplane, and the optimization problem chooses $\beta_0, \beta_1, ..., \beta_p$ to maximize $M$.

### The Non-separable Case

The maximal margin classifier is a very natural way to perform classification, _if a separating hyperplane exists_. We can extend the concept of a separating that _almost_ separates the classes, using a so-called _soft margin_. The generalization of the maximal margin classifier to the non-separable case is known as the _support vector classifier_.

___

## Support Vector Classifiers

### Overview of the Support Vector Classifier

We might be willing to consider a classifier based on a hyperplane that does _not_ perfectly separate the two classes, in the interest of

- greater robustness to individual observations, and
- better classification of _most_ of the training observations.

That is, it could be worthwhile to misclassify a few training observations in order to do a better job in classifying the remaining observations.

The _support vector classifier_, sometimes called a _soft margin classifier_, does exactly this. Rather than seeking the largest possible margin so that every observation is not only on the correct side of the margin, we instead allow some observations to be on the incorrect side of the margin, or even the incorrect side of the hyperplane.

_An observation can be not only on the wrong side of the margin, but also on the wrong side of the hyperplane._

### Details of the Support Vector Classifier

The support vector classifier is the solution to the optimization problem

1. $maximize_{\beta_0, \beta_1, ..., \beta_p, \epsilon_1, ..., \epsilon_n}M$
2. subject to $\sum^p_{j=1}\beta_j^2 = 1$
3. $y_i(\beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + ... + \beta_px_{ip}) \geq M(1 - \epsilon_i)$
4. $\epsilon_i \geq 0, \sum^n_{i=1} \epsilon_i \leq C$

where $C$ is a nonnegative tuning parameter. In (3), $\epsilon_1, ..., \epsilon_n$ are _slack variables_ that allow individual observations to be on the wrong side of the margin or the hyperplane. The slack variable $\epsilon_i$ tells us where the $i^{th}$ observation is located, relative to the hyperplane and relative to the margin. If $\epsilon_i=0$ then the $i^{th}$ observation is on the correct side of the margin, as we saw in (3). If $\epsilon_i>0$ then the $i^{th}$ observation is on the wrong side of the margin, and we say that the $i^{th}$ observation has _violated_ the margin. If $\epsilon_i > 1$ then it is on the wrong side of the hyperplane.

We can think of $C$ as a _budget_ for the amount that the margin can be violated by the $n$ observations. As with the tuning parameters that we have seen, $C$ controls the bias-variance trade-off of the statistical learning technique. When $C$ is small, we seek narrow margins that are rarely violated. On the other hand, when $C$ is larger, the margin is wider and we allow more violations to it; this amounts to fitting the data less hard and obtaining a classifier that is potentially more biased but may have lower variance.

Observations that lie directly on the margin, or on the wrong side of the margin for their class, are known as _support vectors_. These observations do affect the support vector classifier. When the tuning parameter $C$ is large, then the margin is wide, many observations violate the margin, and so there are many support vectors. If $C$ is small, then there will be fewer support vectors and hence the resulting classifier will have low bias but high variance.

___

## Support Vector Machines

### Classification with Non-linear Decision Boundaries

The support vector classifier is a natural approach for classification in the two-class setting, if the boundary between the two classes is linear. However, in practice we are sometimes faced with non-linear class boundaries. In the case of the support vector classifier, we could address the problem of possibly non-linear boundaries between classes in a similar way, by enlarging the feature space using quadratic, cubic, and even higher-order polynomial functions of the predictors. The support vector classifier in a way that leads to efficient computations.

### The Support Vector Machine

The _support vector machine_ (SVM) is an extension of the support vector classifier that results from enlarging the feature space in a specific way, using _kernels_. It turns out that the solution to the support vector classifier problem involves only the _inner products_ of the observations (as opposed to the observations themselves). The inner product of two r-vectors $a$ and $b$ is defined as $< a,b > = \sum^r_{i=1}a_ib_i$. Thus the inner product of two observations $x_i, x_{i'}$ is given by

$< x_i, x_{i'} > = \sum^p_{i=1} x_{ij}x_{i'j}$

It can be shown that 

- the linear support vector classifier can be represented as 
    $f(x) = \beta_0 + \sum^n_{i=1}a_i < x, x_i >$
    where there are $n$ parameters $a_i, i=1, ..., n$, one per training observation.
- to estimate the parameters $a_1, ..., a_n$ and $\beta_0$, all we need are the $(^n_2)$ inner products $< x_i, x_{i'} >$ between all pairs of training observations. (The notation $(^n_2)$ means $n(n-1)/2$, and gives the number of pairs among a set of $n$ items.)

In representing the linear classifier $f(x)$, and in computing its coefficients, all we need are inner products.

Now suppose that every time the inner product $< x_i, x_{i'} > = \sum^p_{j=1} x_ix_{i'j}$ appears in the representation

$f(x) = \beta_0 + \sum^n_{i=1} a_i < x, x_i >$

or in a calculation of the solution for the support vector classifier, we replace it with a _generalization_ of the inner product of the form $K(x_i, x_{i'})$, where $K$ is some function that we will refer to as a _kernel_. A kernel is a function that quantifies the similarity of two observations.

A _polynomial kernel_ of degree $d$, where $d$ is a positive integer and $d>1$, used in the support vector classifier algorithm leads to a much more flexible decision boundary. When the support vector classifier is combined with a non-linear kernel, the resulting classifier is known as a _support vector machine_.

Popular kernel choices of kernels can take the following forms:

- Polynomial kernel
    $K(x_i, x_{i'}) = (1 + \sum^p_{i=1}x_{ij}x_{i'j})^d$
- Radial kernel
    $K(x_i, x_{i'}) = exp(-\gamma \sum^p_{i=1}(x_{ij} - x_{i'j})^2)$

___

## SVMs with More than Two Classes

Separating hyperplanes upon which SVMs are based does not lend itself naturally to more than two classes. Two popular methods for dealing with this are _one-versus-one_ and _one-versus-all_.

### One-versus-one Classification

A _one-versus-one_ or _all-pairs_ approach constructs $(^k_2)$ SVMs, each of which compares a pair of classes. The final classification is performed by assigning the test observation to the class to which it was most frequently assigned in these $(^k_2)$ pairwise classifications.

### One-versus-all Classification

In the _one-versusall_ approach, we fit $k$ SVMs, each time comparing one of the $K$ classes to the remaining $K-1$ classes. We assign the observation to the class for which $\beta_{0k} + \beta_{1x^*_1} + \beta_{2kx^*_2} + ... + \beta_{pkx^*_p}$ is the largest, as this amounts to a high level of confidence that the test observation belongs to the $k^{th}$ class rather to any of the other classes.

___

## Relationship to Logistic Regression

As SVMs have become better understood, deep connections with other more classical statistical methods have emerged. An interesting characteristic of the support vector classifier is that only support vectors play a role in the classifier obtained; observations on the correct side of the margin do not affect it. Due to the similarities between their loss functions, logistic regression and support vector classifier often give very similar results. When the classes are well separated, SVMs tend to behave better than logistic regression; in more overlapping regimes, logistic regression is often preferred.

The choice of the tuning parameter $C$ is very important and determines the extent to which the model underfits or overfits the data.

It is very possible to use a non-linear kernel in logistic regression or many of the other classification methods. However, for historical reasons, the use of non-linear kernels is much more widespread in the context of SVMs then in the context of logistic regression or other methods.