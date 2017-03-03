# Random Forests

_Random forests_ is an ensemble algorithm that functions by training multiple decision trees and combining the results, by voting in the case of classification or by the mean of the outputs in the case of regression. Random forests uses bagging to generate many datasets, but takes the efforts to avoid overfitting one step further by selecting a random subset of features at each split. The introduction of randomization in the data by the bagging process and in the features in the splitting process result in a very robust learner.

Random forests can be applied to classification (Random Forest Classifier) and regression (Random Forest Regressor).

```{python}
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
```

```{r}
library(randomForest)

rfc = randomForest(y~., data=df)
y.pred = predict(rfc, X.test)
```

## Strengths & Weaknesses

Random forests perform quite well modeling counts, rational differences, and ratios. It performs rather poorly on differences, logarithmic functions, polynomials, power functions, rational polynomials, and square root functions. It is also a poor choice for sparse, high dimensional data.

## References

- [An Empirical Analysis fo Feature Engineering for Predictive Modeling](https://arxiv.org/pdf/1701.07852.pdf)