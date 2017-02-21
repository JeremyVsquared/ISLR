# Support Vector Machines

Support vector machines can be applied to regression (_Support Vector Regressor_) or classification (_Support Vector Classifier_).

```{python}
from sklearn.svm import SVR

svr = SVR(C=1.2, epsilon=0.2)
svr.fit(X_train, y_train)

y_pred = svr.predict(X_test)
```

## Strengths & Weaknesses

SVRs performs quite well modeling quadratic functions, rational differences, and ratios. It performs quite poorly on rational polynomials and differences.

## References

- [An Empirical Analysis fo Feature Engineering for Predictive Modeling](https://arxiv.org/pdf/1701.07852.pdf)