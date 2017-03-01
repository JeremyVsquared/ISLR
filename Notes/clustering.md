_Clustering_ or _cluster analysis_ is a form of unsupervised learning that attempts to find subsets or groups within data. In order to accomplish this, the algorithm must be given a definition of similarity or dissimilarity which is supplied by the user. There are a variety of methods to refine this input, but ultimately it must come from outside the data.

There are three general types of clustering algorithms: _combinatorial algorithms_, _mixture modeling_, and _mode seeking_.

_Combinatorial algorithms_ work with the data as it is presented without any assumption of an underlying probability model.

_Mixture modeling_ presumes the data is derived from a combination of component density functions, each component density being a cluster which are fit by way of maximum likelihood or Bayesian methods.

Like mixture modeling, _mode seekers_ presumes an unknown probability density function but differs by attempting to assign clusters by some measure of distance from the centroid.

# K-means

_K-means_ is a combinatorial algorithm, and probably the most popular clustering algorithm. It uses Euclidean distance as its dissimilarity metric and produces $K$ distinct clusters by minimizing the within cluster variation. The user must specify $K$.

```{python}
from sklearn.cluster import KMeans

km = KMeans(n_clusters=8)
km.fit(X_train)
print(km.cluster_centers_)

km.predict(X_test)
```

```{r}
km.cluster = kmeans(df, 8, nstart=20)
km.cluster$cluster
```