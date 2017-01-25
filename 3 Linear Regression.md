# 3 Linear Regression

A linear regression function, at it's most basic, takes the following form

$Y\approx\beta_{0}+\beta_{1}x$

$\approx$ is to mean "is approximately modeled as". A common phrasing of this formula is "regressing Y on X". $\beta_0$ and $\beta_1$ are the coefficients or _weights_, and are constant values. $Y$ is the predicted value and $x$ is the independent, input value. After the coefficients have been trained, the formula can be used to predict new values by

$\hat{y}=\hat{\beta_0}+\hat{\beta_1}x$

where $\hat{y}$ represents a prediction of Y on the basis of X=x.

### Estimating the Coefficients

The formulas coefficients $\beta_0$ and $\beta_1$ are initially unknown, so these must be trained before the algorithm can be used to predict values. The goal of training is to achieve $y_i\approx\hat{\beta_0}+\hat{\beta_{1}}x_1$ for $i=1,...,n$ and a good fit of the model will result in this approximation being relatively close to the observed value. The "closeness" can be measured in a number of ways, but the most common is minimizing the least squares.

Assuming $\hat{y_i}=\hat{\beta_0}+\hat{\beta_1}x_i$, then $e_i=y_i-\hat{y_i}$, in other words the difference between the predicted value and the observed value. $e_i$ is the i<sup>th</sup> _residual_. The residual sum of squares (RSS) as

$RSS=e^2_1+e^2_2+e^2_3+...+e^2_n$

Using a bit of calculus, $\beta_0$ and $\beta_1$ can be minimized by

$\hat{\beta_1}=\frac{\sum^n_{i=1}(x_i-\bar{x})(y_i-\bar{y})}{\sum^n_{i=1}(x_i-\bar{x})^2}$

$\hat{\beta_0}=\bar{y}-\hat{\beta_1}\bar{x}$

where

$\bar{x}=\frac{1}{n}\sum^n_{i=1}x_i$

$\bar{y}=\frac{1}{n}\sum^n_{i=1}y_1$

_y_ and _x_ are the sample means. These are the _least squares coefficient estimates_.

### Assessing the Accuracy of the Coefficients Estimates

We assume that the true relationship between X and Y takes the form $Y=f(X)+\epsilon$ for some unknown $f$, where $\epsilon$ is a mean-zero random error term. $f$ approximated as a linear function would be

$Y=\beta_0+\beta_1X+\epsilon$

$\beta_0$ is the intercept term and $\beta_1$ is the slope.

If we estimate $\hat{\mu}$ based upon the observations in the data set, it is reasonable to assume that given a large enough data set $\hat{\mu}\approx\mu$. The difference between $\hat{\mu}$ and $\mu$ is called the _standard error_ and can be calculated by

$Var(\hat{\mu})=SE(\hat{\mu})^2=\frac{\sigma^2}{n}$

where $\sigma$ is the standard deviation of each of the observations $y_i$ of Y. Note from the above formula that the deviation shrinks as the number of observations $n$ increases.

The standard errors associated with $\hat{\beta_0}$ and $\hat{\beta_1}$ can be calculated by

$SE(\hat{\beta_0})=\sigma^2[\frac{1}{n}+\frac{\bar{x^2}}{\sum^n_{i=1}(x_i-\bar{x})^2}]$

$SE(\hat{\beta_1})^2=\frac{\sigma^2}{\sum^n_{i=1}(x_i-\bar{x})^2}$

where $\sigma^2=Var(\epsilon)$. Note $SE(\hat{\beta_1})$ is smaller when $x_i$ is more spread out. $/sigma^2$ generally is unknown but can be estimated from the data and this estimate of $\sigma$ is known as the _residual standard error_ and is calculated by

$\sigma=RSE=\sqrt{\frac{RSS}{n-2}}$

Standard errors can be used to calculate _confidence intervals_. Ex, a 95% confidence interval is defined as a range such that with 95% probability, the range will contain the true unknown value of the parameter.

For linear regression, the 95% confidence interval for $\beta_0$ and $\beta_1$ are approximately defined by

$\hat{\beta_1}\pm2*SE(\hat{\beta_1})$

$\hat{\beta_0}\pm2*SE(\hat{\beta_0})$

Standard errors are also used to perform hypothesis test, most notably the _null hypothesis_, which sets $\beta_1=0$ to test for having no relationship between X and Y. The null hypothesis reduces the linear regression algorithm to

$y=\beta_0+\epsilon$

If $SE(\hat{\beta_1})$ is small, then $\beta_1\neq0$ and there is a relationship; if $SE(\hat{\beta_1})$ is large, then $\hat{\beta_1}$ must be large in absolute value in order for us to reject the null hypothesis.

The t-statistic measures the number of standard deviations that $\hat{\beta_1}$ is away from 0. This is calculated by

$t=\frac{\hat{\beta_1}-0}{SE(\hat{\beta_1})}$

The probability of observing any value equal to $|t|$ or larger, assuming $\beta_1=0$ is called the p-value. A small p-value indicates that it is unlikely to observe such a substantial association between the predictor and the response due to chance.

P-value is the probability of getting the given results under the condition that the null hypothesis is true. Ie, a p-value of 2% indicates there is only a 2% chance of observing the given values if there was indeed no association. A typical p-value cutoff is between 1% and 5%.

### Assessing the Accuracy of the Model
Once the null hypothesis has been rejected, it is typical to quantify the extent to which the model fits the data. This is usually done with the _residual standard error (RSE)_ and the _$R^2$ statistic_.

#### Residual Standard Error
>$RSE=\sqrt{\frac{1}{n-2}RSS}=\sqrt{\frac{1}{n-2}\sum^n_{i=1}(y_i-\hat{y_i})^2}$
The RSE is an estimate of the standard deviation of $\epsilon$, and is to be considered a measure of lack of fit to the data.

#### R Statistic
> $R^2=\frac{TSS-RSS}{TSS}=1-\frac{RSS}{TSS}$
> $TSS = \sum(y_i-\bar{y})^2$

> Since RSE is measured in the units of Y, it is not always clear what constitutes a good RSE. $R^2$ statistic is a proportion and so is always between 0 and 1. The total sum of squares (TSS) measures the total variance in the response Y.

> _TSS measures the variability in the response before the regression is performed; RSS measures the variability left unexplained after the regression. TSS-RSS measures the variability removed by performing the regression; $R^2$ measures the proportion of variability in Y that can be explained using X._

> High $R^2$ (close to 1) indicates that most variability is explained by the regression; low $R^2$ (close to 0) indicates the regression did not explain much. Low $R^2$ shows the model is wrong or the inherent error $\sigma^2$ is high.

___

## Multiple Linear Regression

$Y=\beta_0+\beta_1x_1+\beta_2x_2+...+\beta_px_p+\epsilon$

Where $x_j$ represents the $j^{th}$ predictor and $\beta_j$ quantifies the association between that variable and the response. $\beta_j$ is interpreted as the average effect on Y of a one unit increase in $x_j$, holding all other predictors fixed.

### Estimating the Regression Coefficients
As with simple linear regression, the parameters $\beta_p$ are unknown, they must be estimated and can be done with the same methods.

$RSS=\sum^n_{i=1}(y_i-\hat{y_i})^2$

$=\sum^n_{i=1}(y_i-\hat{\beta_{0}}-\hat{\beta_1}x_{i1}-...-\hat{\beta_p}x_{ip})^2$

Testing:
Null hypothesis

$H_0=\beta_1=\beta_2=...=\beta_p=0$

and this can be performed by computing the F-statistic

$F=\frac{(TSS-RSS)/p}{RSS/(n-p-1)}$

If the linear model assumptions are correct, it can be shown that

$E\{\frac{RSS}{n-p-1}\}=\sigma^2$

and that, provided $H_0$ is true,

$E\{\frac{TSS-RSS}{p}\}=\sigma^2$

_So when there is no relationship between the response and predictors, one would expect the F-statistic to take on a value close to 1.0. If there is a relationship, that is $H_0$ is false and $H_a$ is true, you would expect F to be greater than 1.0._

1. Is there a relationship?
    When n is large, an F-statistic that is just a little larger than 1 might still provide evidence against $H_0$. However, a larger F-statistic is needed to reject $H_0$ if n is small. This is intuitive as a slight relationship apparent in a small sample is not very convincing.
    
    T-statistics and p-values pertain to individual predictors whereas the f-statistic pertains to the overall model and adjusts for the size of p, but is not a reliable metric of the relationship when $p>n$.
    
    _The first step in multiple regression analysis is to compute the F-statistic and to examine the associated p-value._
2. Deciding on important variables
    Try out a variety of models using a variety of predictor subsets and configurations. There are $2^p$ models that contain subsets of p variables, so exploring all variations rapidly becomes impractical as p increases in size. There are 3 classical approaches:
    1. _Forward selection_: begin with null model (intercept and no predictors), then fit p simple linear regressions and add the model with the lowest RSS to the null model, then add the best performing of a new round of two variable models (the original plus each of p-1 remaining), and continue until some stopping rule is satisfied.
    2. _Backward selection_: begin with all variables and remove the one with the largest p-value (the variable that is least statistically significant), and repeat until some stopping rule is satisfied.
    3. _Mixed selection_: combination of forward and backward; begin with null model, add the variables one by one that provide the best fit, continuously evaluating the p-values of the variables in the model and removing those that cross a set threshold; this forward and backward stepping is continued until all variables in the model have a sufficiently low p-value.
    
    _Backward selection cannot be used if $p>n$; forward selection can always be used._
3. Model fit
    Two of the most common numerical measures of model fit are RSE and $R^2$. An $R^2$ value close to 1 indicates that the model explains a large part of the variance. $R^2$ will always increase when variables are added to the model. When this increases is small, it's probably safe to assume the newly added variable doesn't contribute much to the model fit and could lead to overfitting.

    RSE for multiple linear regression is defined as
    $RSE=\sqrt{\frac{1}{n-p-1}RSS}$
    Therefore it is possible to have higher RSE after a decrease in RSS if there is an increase in p.
4. Predictions
    There are 3 kinds of uncertainty with predictions:
    1. The estimated coefficients $\hat{\beta_0}, \hat{\beta_1},..., \hat{\beta_p}$ are just estimates; _the least squares plan (estimate) is unlikely to actually be the true population regression plane (true value)_.
    2. A linear model $f(x)$ will always contain some degree of _model bias_, which means there will be some amount of irreducible error.
    3. Due to the irreducible error $\epsilon$, _it should never be assumed to be possible to accurately predict y_, even when the true coefficient values are known.

___

## Other Considerations in the Regression Model

### Qualitative Predictors
An easy way of incorporating a classification would be to include a dummy variable with only discrete values. Ie, 

$x_i\begin{cases}1\\0\end{cases}$

$y_i=\beta_0+\beta_1x_i+\epsilon_i=\begin{cases}\beta_0+\beta_1+\epsilon_i\\\beta_0+\epsilon_i\end{cases}$

In this way, $\beta_0$ can be interpreted as a predictor for one class, $\beta_0+\beta_1$ being the average predictor for the other, and $\beta_1$ is the difference between the two. When more than 2 classes need to be considered, you can add to the above with more coefficients.

$y_i=\beta_0+\beta_1x_{i1}+\beta_2x_{i2}+\epsilon_i=\begin{cases} \beta_0+\beta_1+\epsilon_i \\ \beta_0+\beta_2+\epsilon_i \\ \beta_0+\epsilon_i \end{cases}$

In this case, the factor with no dummy variable $(\beta_0+\epsilon_i)$ is referred to as the baseline.

### Extensions of the Linear Model
Two important assumptions are made of the relationship between the predictors and response in linear regression.

1. _Additive_: the effect of changes to a predictor $X_j$ on the response $Y$ is independent of the values of the other predictors.
2. _Linear_: the change in the response $Y$ due to a one-unit change in $X_j$ is constant, regardless of the value of $X_j$.

#### Removing the Additive Assumption
The marketing term "synergy" effect and the statistics term "interaction" effect refer to the same concept of the various real-world conditions interact in ways that may not be readily modeled by a linear regression fit. Considering the standard model

$Y=\beta_0+\beta_1X_1+\beta_2X_2+\beta_3X_1X_2+\epsilon$

The _hierarchical principle_ states that if we include interactions in a model, we should also include the main effects, even if the p-values associated with their coefficients are not significant. In other words, from the above, if the $x_2$ term is not significant but the interaction term is, $x_2$ should still be included in the model.

#### Non-linear Problems
The linear model assumes a linear relationship between the response and predictors, but sometimes this relationship is nonlinear. This can be modeled using _polynomial regression_. This involves using quadratic models and including features multiple times.

### Potential Problems
1. Non-linearity of the data
2. Correlation of error terms
3. Non-constant variance of error terms
4. Outliers
5. High-leverage points
6. Collinearity

___

## Comparison of Linear Regression with K-Nearest Neighbors
_Linear regression is a parametric method_, which has the advantages of being easily interpreted and easily evaluated in accuracy. But it also has the disadvantage of making strong assumptions about the form of $f(x)$; more specifically, it assumes a linear relationship.

_kNN is a non-parametric method_ which does not explicitly assume a parametric form for $f(x)$. kNN regression identifies the k training observations that are closest to $x_0$, represented by $N_0$, and uses these to estimate $f(x_0)$ using the average of all the training responses in $N_0$. This is represented by

$\hat{f}(x_0)=\frac{1}{n}\sum_{x_i \in N_0}y_i$

The value chosen for $k$ impacts the bias-variance tradeoff. Smaller $k$ results in the most flexible fit with low bias and high variance; larger $k$ provides a smoother, less variable fit.

_The parametric approach will outperform the non-parametric approach if the parametric form that has been selected is close to the true form of $f$_. As a general rule, parametric methods will tend to outperform non-parametric approaches when there is a small number of observations per predictor.