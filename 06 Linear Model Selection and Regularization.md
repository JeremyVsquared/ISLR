# 6 Linear Model Selection and Regularization

Here we will discuss some ways in which linear model can be improved, by replacing plain least squares fitting with some alternative fitting procedures.

Alternative fitting procedures can yield better _prediction accuracy_ and _model interpretability_.

- _Prediction accuracy_: Accuracy can be profoundly influenced by the ratio of $n$ to $p$. As such, by _constraining_ or _shrinking_ the estimated coefficients, we can often substantially reduce the variance at the cost of a negligible increase in bias. This can lead to substantial improvements in the accuracy with which we can predict the response for observations not used in model training.
- _Model interpretability_: Coefficients can approach 0 but will very liekly never actually reach 0, indicating an absolute irrelevance to the response value. These apparently irrelevant features can unnecessarily complicate the model and removing them is ideal. There are methods of automatically performing _feature selection_ or _variable selection_ that will exclude these variables from the model.

There are many alternatives to using least squares to fit. The three most important methods are:

1. _Subset selection_: identifying a subset of the $p$ predictors that we believe to be related to the response.
2. _Shrinkage_: fitting a model involving all $p$ predictors but utilizing _regularization_ which could be used to reduce the least relevant coefficients to 0, or near 0.
3. _Dimension reduction_: _projecting_ the $p$ predictors into a $M$-dimensional subspace, where $M\lt p$. This is achieved by computing $M$ different _linear combinations_, or _projections_, of the variables. Then these $M$ projections are used as predictors to fit a linear regression mdoel by least squares.

___

## Subset Selection

### Best Subset Selection

_Best subset selection_ is fitting a separate least squares regression for each possible combination of the _p_ predictors. Selecting the best subset from the $2^p$ possible is broken up into two stages:

1. Let $M_0$ denote the _null model_, which contains no predictors. The model simply predicts the sample mean for each observation.
2. For $k=1, 2, ..., P$:
    a. Fit all $(^{p}_{k})$ models that contain exactly $k$ predictors
    b. Pick the best among these $(^{p}_{k})$ models, and call it $M_k$ is defined as having the smallest RSS, or equivalently largest $R^2$
3. Select a single best model from among $M_0, ..., M_p$ using cross-validated prediction error, $C_p(AIC)$, $BIC$, or adjusted $R^2$.

Step 3 uses cross-validation for model selection by evaluating test error because the RSS of the models decreases monotonically, and the $R^2$ increases monotonically, as the number of features included increases. If these statistics are used to select a model, the model including all features will always be selected. Low RSS or a high $R^2$ indicate a low training error, whereas we are interested in a low test error.

Best subsest selection becoems computationally infeasible for large values of $p$. Additionally, with large $p$ comes a large search space which inherently increases the chances of finding an overfitted model with a low training error but poor predictive performance on future data.

### Stepwise Selection

Due to the inherent weaknesses in best subset selection, it can be advantageous to try an alternative with a more restricted set of possible models.

#### Forward Stepwise Selection

The forward stepwise selection algorithm:

1. Let $M_0$ denote the _null_ model, which contains no predictors.
2. For $k=0, ..., p-1$:
    a. Consider all $p-k$ models that augment the prediction with one additional predictor.
    b. Choose the _best_ among these $p-k$ models, and call it. Here _best_ is defined as having smallest $RSS$ or highest $R^2$.
3. Select a single best model from among $M_0, ..., M_p$ using cross-validated prediction error, $C_p\ (AIC)$, $BIC$, or $adjusted\ R^2$.

Forard stepwise selection begins with the _null_ model and progressively adds features, selecting by the basis of which feature adds the greatest improvement to the fit. While this is more computationally efficient than best subset selection, it is not guaranteed to find the best fit.

#### Backward Stepwise Selection

Backward stepwise selection effectively functions as a reverse forward stepwise selection, starting with all available features and iteratively removing the least useful predictor.

The backward stepwise selection algorithm:

1. Let $M_p$ denote the _full_ model, which contains all $p$ predictors.
2. For $k=p, p-1, ..., 1$:
    a. Consider all $k$ models that contain all but one of the predictors in $M_k$, for a total of $k-1$ predictors.
    b. Choose the _best_ among these $k$ models, and call it $M_k$. Here the _best_ is defined as having smallest $RSS$ or highest $R^2$.
3. Select a single best model from among $M_0, ..., M_p$ using cross-validated predcition error, $C_p\ (AIC)$, $BIC$, or $adjusted\ R^2$.

Like forward stepwise selection, backward stepwise selection is not guaranteed to find the best model, but differs in that backward stepwise selection requires than $n>p$ in order to be applied.

#### Hybrid Approaches

There are hybrid methods that will seemingly alternate between the forward and backward stepwise selection methods. This will effectively run the forward stepwise selection for a few rounds and then switch to backward stepwise selection by removing a few of the least useful predictors.

### Choosing the Optimal Model

Choosing the best mdoel is not always a readily apparent task as a model with all of the predictors will always have the lowest $RSS$ and the largest $R^2$, since these quantities are related to the training error. Selecting the best model is dependent upon the test error.

In order to select the best model with respect to the test error, we need to estimate this test error. There are two common approaches:

1. Indirectly estimate test error by making and _adjustment_ to the training error to account for bias due to overfitting.
2. Directly estimate the test error using either validation or cross-validation.

#### $C_p$, $AIC$, $BIC$, and $Adjusted\ R^2$

As has been previously stated, training set $RSS$ and training set $R^2$ cannot be used to select from among a set of models with different numbers of variables, but there are a number of techniques for _adjusting_ the training error for the model. We will review 4 such techniques: $C_p$, _Akaike information criterion (AIC)_, _Bayesian information criterion (BIC)_, and $adjusted\ R^2$.

- $C_P$
    For a fitted least squares model containing $d$ predictors, the $C_p$ estiamte of test $MSE$ is computed using the equation
    
    $C_p=\frac{1}{n}(RSS+2d\hat{\sigma}^2)$
    
    Where $\hat{\sigma}^2$ is an estimate of the variance of the error $\epsilon$, and $2d\hat{\sigma}^2$ functions as a training error penalty to account for the fact that training error understimates test error. The penalty increases as the number of predictors increases. A lower $C_p$ statistic is indicative of a low test $MSE$ and is to be preferred.

- $AIC$
    The $AIC$ is defined as 
    
    $AIC=\frac{1}{n\hat{\sigma}^2}(RSS+2d\hat{\sigma}^2)$
    
    The $AIC$ is defined for a large class of models fit by maximum likelihood. For least squares models, $C_p$ and $AIC$ are proportional to each other.

- $BIC$
    $BIC$ is derived from a Bayesian point of view. $BIC$ is defined by
    
    $BIC=\frac{1}{n}(RSS+log(n)\hat{\sigma}^2)$
    
    Like $C_p$, the $BIC$ will tend to take on a small value for a model with a low test error, and so generally we select the model that has the lowest $BIC$ value.

- $Adjusted\ R^2$
    $Adjusted\ R^2$ is defined as 
    
    $Adjusted R^2=1-\frac{RSS/(n-d-1)}{TSS/(n-1)}$
    
    A large value of $adjusted R^2$ indicates a model with a low test error. The is that once all of the correct variables have been included in the model, adding additional _noise_ variables will lead to only a very small decrease in $RSS$. Thus, the model with the largest $adjusted\ R^2$ will have only correct variables and no noise variables.

#### Validation and Cross-Validation

While the above techniques do a fine job of estimating test error, the test error can be more directly estimated by performing validation or cross-validation.

___

## Shrinkage Methods

We can fit a model containing all $p$ predictors using a technique that _constrains_ or _regularlizes_ the coefficient estimates, or equivalently, that _shrinks_ the coefficient estimates towards 0, which can significantly reduce their variance. The two best known methods of doing this are _ridge regression_ and _lasso_.

### Ridge Regression

Ridge regression is very similar to least squares excepting that it focuses on minimizing a different quantity. Specifically, the ridge regression coefficients $\hat{\beta}^R$ are values that minimize

$\sum^n_{i=1}(y_i - \beta_0 - \sum^p_{j=1}\beta_jx_{ij})^2 + \lambda \sum^p_{i=1}\beta_j^2 = RSS + \lambda \sum^p_{j=1}\beta^2_j$

Where $\lambda\geq0$ is a tuning parameter. This is standard least squares with the penalty term to minimize the magnitude of the coefficients. When $\lambda=0$, the penalty term meant to push coefficients toward $0$ is canceled out, thus having no effect. However, as $\lambda \rightarrow \infty$, the impact of the shrinkage penalty grows, and the ridge regression coefficient estimates will approach $0$.

It is best to us cross-validation for choosing the value of $\lambda$ and apply ridge regression after standardizing the predictors using the following forumala to be sure they're all on the same scale.

$\bar{x}_{ij}=\frac{x_{ij}}{\sqrt{\frac{1}{n}\sum^n_{i=1}(x_{ij}-\bar{x}_j)^2}}$

The denominator above is the estimated standard deviation of the $j^{th}$ predictor.

#### Why Does Ridge Regression Improve Over Least Squares?

The advantage of ridge regression over least squares is that as $\lambda$ increases, the flexibility of the ridge regression fit decreases, leading to decreased variance but increased bias. Thus, ridge regression works best in cases where the least squares estimates have high variance as the $\lambda$ term can be used to control it.

### Lasso

Ridge regresison will always generate a model involving all available predictors, and lasso is an alternative that can overcome this disadvantage. The lasso coeffecients, $\hat{\beta}^2_\lambda$, minimize the quantity

$\sum^n_{i=1}(y_i - \beta_0 - \sum^p_{j=1}\beta_kx_{ij})^2 + \lambda\sum^p_{j=1}|\beta_j| = RSS + \lambda\sum^p_{j=1}|\beta_j|$

In statistical parlance, the lasso uses and $\ell_1$ penalty instead of an $\ell_2$ penalty. The $\ell_1$ norm of a coefficient vector $\beta$ is given by $||\beta||_1=\sum|\beta_j|$.

As with ridge regression, the lasso shrinks the coefficient estimates towards zero. However, in the case of the lasso, the $\ell_1$ penalty has the effect of forcing some of the coefficient estimates to be exactly equal to zero when the tuning parameter $\lambda$ is sufficiently large, consequently performing feature selection.

Comparing lasso to ridge regression, we would expect the lasso to perform better in cases where a relatively small number of predictors have substantial coefficients, and the remaining predictors have coefficients that are very small or equal to zero. Ridge regression will perform better when the response is a function of many predictors, all with coefficients of roughly equal size.

Like ridge regression, when least squares have excessively high variance, the lasso solution can yield a reduction in variance at the expense of a small increase in bias, and consequently can generate more accurate predictions. Unlike ridge regression, lasso performs variable selection, and hence results in models that are easier to interpret.

_Lasso is a convex optimization._

### Selecting the Tuning Parameter

Cross-validation provides a simple way to tackle this problem.

1. We choose a grid of $\lambda$ values,
2. Compute the cross-validation error for each value of $\lambda$
3. Select the tuning parameter $\lambda$ with the lowest cross-validation error
4. Re-fit the model using all available observations and the selected value of the tuning parameter

___

## Dimension Reduction Methods

The above methods have controlled variance by using a subset of the original variables or by shrinking their coefficients. Dimension reduction methods, on the other hand, transform predictors and then fit least squares to the transformed variables. All dimension reduction methods work in two steps:

1. The transformed predictors $Z_1, Z_2, ..., Z_M$ are obtained
2. The model is fit using these $M$ predictors

The choice $Z_1, Z_2, ..., Z_M$ can be achieved in different ways. We will look at two, _principal components_ and _partial least squares_.

### Principal Components Regression

Principal component analysis (PCA) is a popular approach for deriving a low-dimensional set of features from a large set of variables and a technique for reducing the dimension of a $nXp$ data matrix $X$. The _first principal component_ direction of the data is that along which the observations _vary the most_.

The _principal components regression_ (PCR) approach involves constructing the first $M$ principal components, $Z_1, ..., Z_M$, and then using these components as the predictors in a linear regression model that is fit using least squares. The key idea is that often a small number of principal components suffice to explain most of the variability. This is fundamentally dependent upon the assumption that the directions in which $X_1, ..., X_p$ show the most variation are the directions that are associated with $Y$.

PCR will do well in cases when the first few principal components are sufficient to capture most of the variation in the predictors as well as the relationship with the response. Additionally, PCR does not perform feature selection as each of the $M$ principal components used in the regression is a linear combination of all $p$ of the original features. One can think of ridge regression as a continuous version of PCR.

Scaling features prior to performing PCR is recommended for best results.

### Partial Least Squares

The PCR approach identifies linear combination in an unsupervised way. Consequently, there is no guarantee that the directions that best explain the predictors will also be the best directions to use for predicting the response. _Partial least squares_ (PLS) is a supervised alternative to PCR.

Like PCR, PLS is a dimension reduction method, which first identifies a new set of features $Z_1, ..., Z_M$ that are linear combinations of the original features, and fits a linear model via least squares using the $M$ new features. Unlike PCR, PLS identifies these new features in a supervised way, using the response $Y$ in order to identify new features that not only approximate the old features well, but also that are related to the response.

While the supervised dimension reduction of PLS can reduce bias, it also has the potential to increase variance, so that the overall benefit of PLS relative to PCR is a wash.

___

## Considerations in High Dimensions

### High-dimensional Data

Most traditional statistical techniques for regression and classification are intended for the _low-dimensional_ setting in which $n$, the number of observations, is much greater than $p$, the number of features. Data sets containing more features than observations are often referred to as _high-dimensional_. Classical approaches such as least squares linear regression are not appropriate in this setting.

### What Goes Wrong in High Dimensions?

In cases of high dimensionality, where $n>p$, least squares will tend to yield a set of coefficient estimates that result in a perfect fit to the data, such that the residuals are zero regardless of whether or not there truly is a relationship. This does not result in a model in a model fit that performs well on an independent test set due to overfitting. The model $R^2$ increases to 1 as the number of features included in the model increases, and correspondingly the training set MSE decreases to 0 as the number of features increases, _even though the features are completely unrelated to the response_.

### Regression in High Dimensions

Many methods for fitting less flexible least squares models, such as forward stepwise selection, ridge regression, the lasso, and principal components regression, are particularly useful for performing regression in the high-dimensional setting.

A key principal in the analysis of high dimensional data is that the test error tends to increase as the dimensionality of the problem increases unless the additional features are truly associated with the response. The effect is known as the _curse of dimensionality_. In general, adding additional signal features that are truly associated with the response will improve the fitted model, in the sense of leading to a reduction in test set error. However, adding noise features that are not truly associated an increased test set error. This is because noise features increase the dimensionality of the problem, exacerbating the risk of overfitting.

### Interpreting Results in High Dimensions

When working with high dimensional problems, one must be especially cautious of _multicollinearity_, the concept that the variables in a regression might be correlated with each other.

When _p>n_, it is easy to obtain a useless model that has 0 residuals. Therefore, one should never use sum of squared errors, p-values, $R^2$ statistics, or other traditional measures of model fit on the training data as evidence of a good model fit in high-dimensional setting.