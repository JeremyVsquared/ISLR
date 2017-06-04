# Nearest Neighbors

_K-nearest neighbors_ (KNN) is a learning algorithm with a fairly simple premise of averaging the target of the observations that are nearest the predictor. The value of $K$ determines the number of neighbors used to generate the prediction. $K$ dictates the bias-variance tradeoff: large $K$ will generate a smoother fit with low variance and high bias, small $K$ will generate a low bias but high variance fit. Additionally, large $K$ can result in difficulties in fit due to dimensionality as it becomes increasingly difficult to locate suitably close neighbors. This closeness is most often evaluated by Euclidean distance. In this case, feature extraction and/or dimensionality reduction can be helpful to resolve the issue. The choice of $K$ should be made based upon cross-validation.

One of the major benefits of KNN is that it is non-parametric and relies upon no assumptions about the distribution of the underlying data as a linear method does. Additionally, KNN does not _train_ the way a parametric method would, rather computation does not take place until prediction.

KNN can be applied to classification by applying the most common label of the nearest neighbors or regression by averaging the target of the nearest neighbors.

A potential problem with KNN's are class outliers, or observations that fall outside of normal ranges and are surrounded by observations with another label. This can be caused by errors in the data or a _class imbalance_. Class imbalance is the presence of an overwhelming majority of a particular class, such as 90% class A and 10% class B. These circumstances can be difficult to overcome as they often lead to incorrect assumptions about the data or invalid predictions.

```{python}
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
```

```{r}
y.pred = knn(X.train, X.test, y.train, k=10)
```

## Extending Nearest Neighbors

One of the greatest weaknesses of KNN is it's performance suffers with sparse or tightly clustered, overlapping class data. There are numerous, less studied algorithms that can be proven to perform better than traditional KNN which aim to make KNN more flexible in these troublesome circumstances by adapting $k$ to observational circumstances. These algorithms typically apply a sliding scale of $k$ rather than a constant value to better classify a given observation by discriminant metrics such as comparison of the Euclidean distance to neighbors. While less tested and less well known, research in this area has concluded that many of these methods perform better than traditional KNN.

- http://www.cs.toronto.edu/~cuty/LEKM.pdf
- http://sci2s.ugr.es/keel/pdf/algorithm/articulo/knnAdaptive.pdf
- http://delab.csd.auth.gr/papers/ADBIS07onpmw.pdf
