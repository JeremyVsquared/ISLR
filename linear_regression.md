# Regression

_Regression_ algorithms predict continuous values as opposed to _classification_ algorithms which predict discrete values. Predicting weather offers an example of this. Regression would be used to predict inches of rain, classification would be used to predict whether or not it will rain. Geometrically, the difference could be thought of in terms of optimized Euclidean distance. An optimized, well-fitted regression algorithm will minimize the average Euclidean distance between a predicted line and the observations. A similarly optimized, well-fitted classification algorithm would maximize this distance.

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


## Feature selection

Multivariate linear regression can be refined by _feature selection_ as not all features $x_p$ will postively contribute to fit. It is often useful to examine the features being used for a model to verify that a better fit could not be achieved by a subset of the features. This can be done by _best subset_, _forward stepwise_ or _backward stepwise_ selection.

_Best subset_ selection is the process of trying all possible subsets of the features $x_p$. This is effective, but inefficient as it trains a model for all permutations of the feature set. _Forward stepwise_ selection starts with no features, only a constant, and then progressively adds the feature that most positively contributes to the fit. _Backward stepwise_ selection works much like forward stepwise but in reverse. It starts with all features and progressively removes the feature with the least impact on the fit.

## Shrinkage

An alternative methodology to _feature selection_ is _shrinkage_, which performs a similar function of reducing the impact of weak or negatively contributing features with a different strategy. Where feature selection removes features, shrinkage reduces the impact of these features by minimizing or setting to 0 the corresponding coefficient. Shrinkage algorithms include _Lasso_ and _Ridge Regression_.

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