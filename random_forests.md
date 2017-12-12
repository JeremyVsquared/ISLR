# Random Forests

_Random forests_ is an ensemble algorithm that functions by training many decision trees and combining the results, by voting in the case of classification (mode) or by the averaging the outputs (mean) in the case of regression. Random forests use bagging to generate many datasets, but takes the effort to avoid overfitting one step further by selecting a random subset of features at each split. The introduction of randomization in the data by the bagging process and in the feature selection in the splitting process result in a very robust learner by decorrelating the trees.

Random forests can be applied to classification (Random Forest Classifier) and regression (Random Forest Regressor).

```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
```

```r
library(randomForest)

rfc = randomForest(y~., data=df)
y.pred = predict(rfc, X.test)
```

Random forests perform better than decision trees by avoiding overfitting by averaging the results of many trees, decreasing variance by the random selection of features per tree, and decreasing bias by doing all of this with bagged decision trees.

## Strengths & Weaknesses

Random forests perform quite well modeling counts, rational differences, and ratios. It performs rather poorly on differences, logarithmic functions, polynomials, power functions, rational polynomials, and square root functions. It is also a poor choice for sparse, high dimensional data.

## References

- [An Empirical Analysis fo Feature Engineering for Predictive Modeling](https://arxiv.org/pdf/1701.07852.pdf)