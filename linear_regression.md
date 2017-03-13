# Regression

_Regression_ algorithms predict continuous values as opposed to _classification_ algorithms which predict discrete values. Predicting weather offers an example of this. Regression would be used to predict inches of rain, classification would be used to predict whether or not it will rain. Geometrically, the difference could be thought of in terms of optimized Euclidean distance. A well-fitted regression will minimize the average Euclidean distance between a predicted line and the observations while a similarly well-fitted classification will maximize this distance.

# Linear Regression

_Linear regression_ functions by iteratively correcting functional parameters in order to minimize the average difference between predicted values and the target values. Geometrically, it generates a predicted line and then moves it until it has minimized the average distance between the observations and the line. In it's simplest, univariate form, it has a single parameter $x$ and a target $y$, but it works equally well in multivariate form with multiple inputs $x_1, x_2, ..., x_p$.

```{python}
from sklearn.linear_model import LinearRegression

mdl = LinearRegression()
mdl.fit(X_train, y_train)

y_pred = mdl.predict(X_test)
```

```{r}
mdl = lm(y~., data=df)
summary(mdl)

y.pred = predict(mdl, df.test)
```

## Ordinary Least Squares

The most common linear regression algorithm is _Ordinary Least Squares_ (OLS) which uses the squared error, or _residual sum of squares (RSS) $\sum_i (y_i - f(x_i))^2$, the difference between the predicted output and the true value, as it's measurement of fit. An optimized OLS fit will minimize the squared error by iteratively adjusting the $\hat{\beta}$ coefficients.


## Feature selection

Multivariate linear regression can be refined by _feature selection_ as not all features $x_p$ will postively contribute to fit. It is often useful to examine the features being used for a model to verify that a better fit could not be achieved by a subset of the features. This can be done by _best subset_, _forward stepwise_ or _backward stepwise_ selection.

_Best subset_ selection is the process of trying all possible subsets of the features $x_p$. This is effective, but inefficient as it trains a model for all permutations of the feature set. _Forward stepwise_ selection starts with no features, only a constant, and then progressively adds the feature that most positively contributes to the fit. _Backward stepwise_ selection works much like forward stepwise but in reverse. It starts with all features and progressively removes the feature with the least impact on the fit.

## Shrinkage

An alternative methodology to _feature selection_ is _shrinkage_, which performs a similar function of reducing the impact of weak or negatively contributing features with a different strategy. Where feature selection removes features, shrinkage reduces the impact of these features by minimizing the corresponding coefficient, even setting it to 0 in some cases. Shrinkage algorithms include _Lasso_ and _Ridge Regression_.

Both ridge regression and the lasso reduce the impact of features by adding a penalty term to coefficients. The ridge regression uses an $L2$ penalty ($\sum^p_1 \beta^2_j$) and the lasso uses an $L1$ penalty ($\sum^p_1 \beta_j$), both with a coefficient of $\lambda$ which is used to moderate the penalty. The $\lambda$ values are $\ge 0$ and are correlated with the degree of shrinkage that occurs; in other words, the larger the value of $\lambda$, the more the feature coefficient will be diminished. In practice, the $\lambda$ value is chosen by cross validation. $\lambda = 0$ will result in ordinary least squares for both ridge and the lasso.

```{python}
from sklearn.linear_model import Ridge

mdl = Ridge(alpha=0.2)
mdl.fit(X_train, y_train)

y_pred = mdl.predict(X_test)
```

```{python}
from sklearn.linear_model import Lasso

mdl = Lasso(alpha=0.1)
mdl.fit(X_train, y_train)

y_pred = mdl.predict(X_test)
```

## Dimensionality Reduction

A high dimensional (ie, has many features) dataset can prove problematic to fit for a variety of reasons from computational limitations to weak fits. One way to resolve these issues is feature selection, but this is not always ideal. Another way to reduce the number of features is to generate linear combinations of the original features and regress on those.

### Principal Component Regression

As the name implies, _principal component regression_ (PCR) regresses on the principal components of the dataset. PCR reduces the dimensionality of the data with the additional benefit of resolving multicollinearity. Additionally, since the greatest proportion of variation and explanation of the features is concentrated within the principal components, PCR will theoretically provide a better fit than an _ordinary least squares_ regression trained on the original data. The number of derived features $M$ should be chosen by cross-validation.

```{r}
library(pls)

pcr.mdl = pcr(y~., data=df.train, scale=TRUE, validation="CV)
summary(pcr.mdl)

y.pred = predict(pcr.mdl, df.test)
```

### Partial Least Squares

_Parial Least Squares_ (PLS) is another dimensionality reducing regression method that, unlike PCR, attempts to find derived features based upon their explanation of the target. PLS accomplishes this by prioritizing projected features that are most correlated with the target. As with PCR, the number of derived features $M$ should be chosen by cross-validation. This method is known to reduce bias, but can also increase variance.

```{r}
library(pls)

pls.mdl = plsr(y~., data=df.train, scale=TRUE, validation="CV")
summary(pls.mdl)

y.pred = predict(pls.mdl, df.test)
```