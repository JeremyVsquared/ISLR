# Gradient Boosting Machines

Gradient Boosting Machines, or GBMs, are a boosted algorithm that generates successive decision tree models, each of which is intended to resolve the errors of the previous model. The most important features to be determined when applying this algorithm are the _number of trees_, which will define the number of models to be generated, and the _learning rate_ or _shrinkage_, which dictates the severity of the corrective action as the GBM attempts to resolve it's past errors.

On the whole, increasing the _number of trees_ will not quickly lead to overfitting and the user may do so until improvement in fit stops.

The _learning rate_ should be small (between $0$ and $1$). Having an extremely small learning rate will lead to slow fits, while large learning rates can lead to divergence. In practice, the learning rate should tend toward being $< 0.3$.

GBMs can be applied to regression (_Gradient Boosting Regressor_) or classification (_Gradient Boosting Classifier_).

```{python}
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=250, learning_rate=0.003, loss='ls')
gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)
```

```{r}
library(gbm)

gbm = gbm(y ~., df, n.trees=250, shrinkage=0.003, distribution="gaussian", interaction.depth=7, bag.fraction=0.9, cv.fold=10, n.minobsinnode = 50)

y.pred = predict(gbm, df_test, type="response")
```

## Strengths & Weaknesses

GBMs perform quite well modeling counts, rational differences, and ratios. It performs poorly with differences, logarithmic functions, polynomials, power functions, and rational polynomials.

## References

- [An Empirical Analysis fo Feature Engineering for Predictive Modeling](https://arxiv.org/pdf/1701.07852.pdf)
