# Gradient Boosted Machines

Gradient boosting machines can be applied to regression (_Gradient Boosting Regressor_) or classification (_Gradient Boosting Classifier_).

```{python}
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=250, learning_rate=0.003, loss='ls')
gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)
```

## Strengths & Weaknesses

Gradient boosted machines perform quite well modeling counts, rational differences, and ratios. It performs poorly with differences, logarithmic functions, polynomials, power functions, and rational polynomials.

## References

- [An Empirical Analysis fo Feature Engineering for Predictive Modeling](https://arxiv.org/pdf/1701.07852.pdf)