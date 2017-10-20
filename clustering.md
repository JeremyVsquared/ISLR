# Clustering

_Clustering_ or _cluster analysis_ is a form of unsupervised learning that attempts to find subsets or groups within data. In order to accomplish this, the algorithm must be given a definition of similarity or dissimilarity which is supplied by the user. There are a variety of methods to refine this input, but ultimately it must come from outside the data.

# K-means

_K-means_ is probably the most popular clustering algorithm. It uses Euclidean distance as its dissimilarity metric and produces $K$ distinct clusters by minimizing the within cluster variation. The user must specify $K$.

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=8)
km.fit(X_train)
print(km.cluster_centers_)

km.predict(X_test)
```

```r
km.cluster = kmeans(df, 8, nstart=20)
km.cluster$cluster
```

## Validation

Applying a clustering algorithm to data will find clusters, whether these clusters are statistically relevant or not. In other words, a cluster analyis of uniformly distributed data with $K$ equal to 2, 5, or 20 will indeed find 2, 5, and 20 clusters despite the data being uniformly distributed. Given this dilemma, it is important to establish a validation metric for a cluster analysis to verify that the clusters are relevant.. The K-means algorithm is fundamentally evaluating the homogeneity or heterogeneity of various subsets of the data. This subset homogeneity and heterogeneity can be used to evaluate the clustering outcomes, or more specifically, the chosen value of $K$. While a consensus has yet to arise on the best method, the two most popular methods of validating choices of $K$ are the _elbow method_ and the _gap statistic_.

### Elbow Method

The _elbow method_ is a graphical evaluation of the _percentage of variance explained_ (PVE). This metric quantifies the relationship, or ratio, between the between-group variance and within-group variance. Plotting this value as a function of $K$ can present a value of $K$ where the the variance explained gain between choices of $K$ distinctly drops which indicates an optimal choice of $K$. This distinct drop in gain is referred to as the "elbow" of the graph. While this is an effective method, it is not always possible to clearly distinguish the ideal value of $K$ when the gain in variance explained between values of $K$ is sufficiently flat.

### Gap Statistic

- the gap statistic compares the curve $log W_K$ to the curve obtatined from data uniformly distributed over a rectangle containing the data; it estimates the optimal number of clusters to be the place where the gap between the two curves is largest
- this is an automatic way of locating the aforementioned "kink" (referene to the elbow method)
- the idea behind their apprach was to find a way to standardize the comparison of $log W_k$ with a null reference distribution of the data, ie, a distribution with no obvious clustering
- their estimate for the optimal number of clusters $K$ is the value for which $log W_k$ falls the farthest below this reference curve
    - $$Gap_n (k) = E^*_n {log W_k} - log W_k$$
    - to obtain the estimate $E^*_n {log W_k}$ we compute the average of $B$ copies $log W^*_k$ for $B=10$, each of which is generated with a Monte Carlo sample from the reference distribution
    - those $log W^*_k$ from the $B$ Monte Carlo replicates exhibit a standard deviation $sd(k)$ which, accounting for the simulation error, is turned into the quantity $s_k = \sqrt{1 + 1/B} sd(k)$
    - the optimal number of clusters $K$ is the smallest $k$ such that $Gap(k) \geq Gap(k + 1) - s_{k + 1}$

# Hierarchical Clustering

- 


- [On Clustering Validation Techniques](http://web.itu.edu.tr/sgunduz/courses/verimaden/paper/validity_survey.pdf)
- [Estimating the numbers of clusters in a data set via the gap statistic](http://www.stanford.edu/~hastie/Papers/gap.pdf)