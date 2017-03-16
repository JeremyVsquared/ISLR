# Nearest Neighbors

_K-nearest neighbors_ (KNN) is a non-parametric learning algorithm with a fairly simple premise of averaging the target of the observations that are nearest the predictor. The value of $K$ determines the number of neighbors used to generate the prediction. $K$ dictates the bias-variance tradeoff: large $K$ will generate a smoother fit with low variance and high bias, small $K$ will generate a low bias but high variance fit. Additionally, large $K$ can result in difficulties in fit due to dimensionality as it becomes increasingly difficult to locate a suitably "close" neighbor. One of the major benefits of KNN is that it relies upon no assumptions about the underlying data as a linear method does.

KNN can be applied to classification by applying the most common label of the nearest neighbors or regression by averaging the target of the nearest neighbors.

```{python}
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
```

```{r}
y.pred = knn(X.train, X.test, y.train, k=10)
```